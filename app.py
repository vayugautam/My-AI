import http.server
import socketserver
import json
import math
import heapq
import random
import threading
import time
import urllib.request
import re
import os
from urllib.parse import urlparse, parse_qs

# =====================================================================
#  DATA TYPES / CONFIGURATION
# =====================================================================

DIMS = 16  # Demo vector dimensions
random.seed(42)

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    val = dot / (math.sqrt(na) * math.sqrt(nb))
    # Clip val to prevent float precision errors causing math.domain error or distance < 0
    val = max(min(val, 1.0), -1.0)
    return 1.0 - val

def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric):
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items = []

    def insert(self, item):
        self.items.append(item)

    def knn(self, q, k, dist_fn):
        r = [(dist_fn(q, item["emb"]), item["id"]) for item in self.items]
        r.sort(key=lambda x: (x[0], x[1]))
        return r[:k]

    def remove(self, item_id):
        self.items = [item for item in self.items if item["id"] != item_id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, dims):
        self.dims = dims
        self.root = None

    def insert(self, item):
        self.root = self._ins(self.root, item, 0)

    def _ins(self, node, item, depth):
        if not node:
            return KDNode(item)
        ax = depth % self.dims
        if item["emb"][ax] < node.item["emb"][ax]:
            node.left = self._ins(node.left, item, depth + 1)
        else:
            node.right = self._ins(node.right, item, depth + 1)
        return node

    def knn(self, q, k, dist_fn):
        if not self.root:
            return []
        heap = []  # Max-heap of (-dist, id)
        self._knn(self.root, q, k, 0, dist_fn, heap)
        r = [(-dist_neg, item_id) for dist_neg, item_id in heap]
        r.sort(key=lambda x: (x[0], x[1]))
        return r

    def _knn(self, node, q, k, depth, dist_fn, heap):
        if not node:
            return
        dn = dist_fn(q, node.item["emb"])
        if len(heap) < k:
            heapq.heappush(heap, (-dn, node.item["id"]))
        elif -dn > heap[0][0]:
            heapq.heapreplace(heap, (-dn, node.item["id"]))

        ax = depth % self.dims
        diff = q[ax] - node.item["emb"][ax]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist_fn, heap)
        max_dist_in_heap = -heap[0][0] if heap else float("inf")
        if len(heap) < k or abs(diff) < max_dist_in_heap:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)

    def rebuild(self, items):
        self.root = None
        for item in items:
            self.insert(item)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    def __init__(self, item, max_lyr):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs = [[] for _ in range(max_lyr + 1)]

class HNSW:
    def __init__(self, m=16, ef_build=200):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.topLayer = -1
        self.entryPt = -1
        self.G = {}  # id -> HNSWNode

    def rand_level(self):
        r = random.random()
        if r == 0:
            r = 1e-9
        return int(math.floor(-math.log(r) * self.mL))

    def search_layer(self, q, ep, ef, lyr, dist_fn):
        vis = {ep}
        d0 = dist_fn(q, self.G[ep].item["emb"])
        cands = [(d0, ep)]  # Min-heap
        found = [(-d0, ep)]  # Max-heap

        while cands:
            cd, cid = heapq.heappop(cands)
            max_found_dist = -found[0][0]
            if len(found) >= ef and cd > max_found_dist:
                break
            
            node = self.G.get(cid)
            if not node or lyr >= len(node.nbrs):
                continue
            
            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist_fn(q, self.G[nid].item["emb"])
                max_found_dist = -found[0][0]
                if len(found) < ef or nd < max_found_dist:
                    heapq.heappush(cands, (nd, nid))
                    if len(found) < ef:
                        heapq.heappush(found, (-nd, nid))
                    else:
                        heapq.heapreplace(found, (-nd, nid))

        res = [(-dist_neg, nid) for dist_neg, nid in found]
        res.sort(key=lambda x: (x[0], x[1]))
        return res

    def select_nbrs(self, cands, max_m):
        return [cid for _, cid in cands[:max_m]]

    def insert(self, item, dist_fn):
        item_id = item["id"]
        lvl = self.rand_level()
        node = HNSWNode(item, lvl)
        self.G[item_id] = node

        if self.entryPt == -1:
            self.entryPt = item_id
            self.topLayer = lvl
            return

        ep = self.entryPt
        for lc in range(self.topLayer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(item["emb"], ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.topLayer, lvl), -1, -1):
            W = self.search_layer(item["emb"], ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self.select_nbrs(W, max_m)
            self.G[item_id].nbrs[lc] = sel

            for nid in sel:
                neighbor = self.G.get(nid)
                if not neighbor:
                    continue
                if len(neighbor.nbrs) <= lc:
                    neighbor.nbrs.extend([[] for _ in range(lc + 1 - len(neighbor.nbrs))])
                conn = neighbor.nbrs[lc]
                conn.append(item_id)

                if len(conn) > max_m:
                    ds = [(dist_fn(neighbor.item["emb"], self.G[c].item["emb"]), c) for c in conn if c in self.G]
                    ds.sort(key=lambda x: (x[0], x[1]))
                    neighbor.nbrs[lc] = [c for _, c in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self.topLayer:
            self.topLayer = lvl
            self.entryPt = item_id

    def knn(self, q, k, ef, dist_fn):
        if self.entryPt == -1:
            return []
        ep = self.entryPt
        for lc in range(self.topLayer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(q, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]
        W = self.search_layer(q, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, item_id):
        if item_id not in self.G:
            return
        for nid, node in list(self.G.items()):
            for layer in node.nbrs:
                if item_id in layer:
                    layer.remove(item_id)
        if self.entryPt == item_id:
            self.entryPt = -1
            for nid in self.G:
                if nid != item_id:
                    self.entryPt = nid
                    break
        del self.G[item_id]

    def get_info(self):
        max_l = max(self.topLayer + 1, 1)
        info = {
            "topLayer": self.topLayer,
            "nodeCount": len(self.G),
            "nodesPerLayer": [0] * max_l,
            "edgesPerLayer": [0] * max_l,
            "nodes": [],
            "edges": []
        }
        for item_id, node in self.G.items():
            info["nodes"].append({
                "id": item_id,
                "metadata": node.item["metadata"],
                "category": node.item["category"],
                "maxLyr": node.max_lyr
            })
            for lc in range(min(node.max_lyr + 1, max_l)):
                info["nodesPerLayer"][lc] += 1
                if lc < len(node.nbrs):
                    for nid in node.nbrs[lc]:
                        if item_id < nid:
                            info["edgesPerLayer"][lc] += 1
                            info["edges"].append({
                                "src": item_id,
                                "dst": nid,
                                "lyr": lc
                            })
        return info

    def __len__(self):
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE
# =====================================================================

class VectorDB:
    def __init__(self, dims):
        self.dims = dims
        self.store = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.lock = threading.Lock()
        self.nextId = 1

    def insert(self, meta, cat, emb, dist_fn):
        with self.lock:
            item_id = self.nextId
            self.nextId += 1
            item = {"id": item_id, "metadata": meta, "category": cat, "emb": emb}
            self.store[item_id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist_fn)
            return item_id

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q, k, metric, algo):
        with self.lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dist_fn)
            else:
                raw = self.hnsw.knn(q, k, 50, dist_fn)

            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, item_id in raw:
                if item_id in self.store:
                    item = self.store[item_id]
                    hits.append({
                        "id": item_id,
                        "metadata": item["metadata"],
                        "category": item["category"],
                        "emb": item["emb"],
                        "distance": d
                    })
            return hits, us, algo, metric

    def benchmark(self, q, k, metric):
        with self.lock:
            dist_fn = get_dist_fn(metric)
            
            t_bf = time.perf_counter()
            self.bf.knn(q, k, dist_fn)
            bf_us = int((time.perf_counter() - t_bf) * 1_000_000)

            t_kd = time.perf_counter()
            self.kdt.knn(q, k, dist_fn)
            kd_us = int((time.perf_counter() - t_kd) * 1_000_000)

            t_hnsw = time.perf_counter()
            self.hnsw.knn(q, k, 50, dist_fn)
            hnsw_us = int((time.perf_counter() - t_hnsw) * 1_000_000)

            return bf_us, kd_us, hnsw_us, len(self.store)

    def all_items(self):
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self):
        with self.lock:
            return self.hnsw.get_info()

    def size(self):
        with self.lock:
            return len(self.store)

# =====================================================================
#  DOCUMENT DATABASE
# =====================================================================

class DocumentDB:
    def __init__(self):
        self.store = {}
        self.bf = BruteForce()
        self.hnsw = HNSW(16, 200)
        self.lock = threading.Lock()
        self.nextId = 1
        self.dims = 0

    def insert(self, title, text, emb):
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            item_id = self.nextId
            self.nextId += 1
            
            doc_item = {"id": item_id, "title": title, "text": text, "emb": emb}
            self.store[item_id] = doc_item
            
            vi = {"id": item_id, "metadata": title, "category": "doc", "emb": emb}
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item_id

    def search(self, q, k, max_dist=0.7):
        with self.lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            
            out = []
            for d, item_id in raw:
                if item_id in self.store and d <= max_dist:
                    out.append((d, self.store[item_id]))
            return out

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all_items(self):
        with self.lock:
            return list(self.store.values())

    def size(self):
        with self.lock:
            return len(self.store)

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text, chunk_words=250, overlap_words=30):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    
    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_words])
        chunks.append(chunk)
        if i + chunk_words >= len(words):
            break
        i += step
    return chunks

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host="127.0.0.1", port=11434):
        self.host = host
        self.port = port
        self.embedModel = "nomic-embed-text"
        self.genModel = "llama3.2"

    def is_available(self):
        url = f"http://{self.host}:{self.port}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as res:
                return res.status == 200
        except Exception:
            return False

    def embed(self, text):
        url = f"http://{self.host}:{self.port}/api/embeddings"
        data = json.dumps({"model": self.embedModel, "prompt": text}).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30.0) as res:
                if res.status != 200:
                    return []
                resp_data = json.loads(res.read().decode("utf-8"))
                return resp_data.get("embedding", [])
        except Exception as e:
            print(f"Error in Ollama embed: {e}")
            return []

    def generate(self, prompt):
        url = f"http://{self.host}:{self.port}/api/generate"
        data = json.dumps({"model": self.genModel, "prompt": prompt, "stream": False}).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=180.0) as res:
                if res.status != 200:
                    return "ERROR: Ollama unavailable. Run: ollama serve"
                resp_data = json.loads(res.read().decode("utf-8"))
                return resp_data.get("response", "")
        except Exception as e:
            print(f"Error in Ollama generate: {e}")
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  HTTP HANDLER & SERVER
# =====================================================================

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

def load_demo_data():
    dist = get_dist_fn("cosine")
    # CS
    db.insert("Linked List: nodes connected by pointers", "cs",
        [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06], dist)
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
        [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10], dist)
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
        [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07], dist)
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
        [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07], dist)
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
        [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09], dist)
    # Math
    db.insert("Calculus: derivatives integrals and limits", "math",
        [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10], dist)
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09], dist)
    db.insert("Probability: distributions random variables Bayes theorem", "math",
        [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08], dist)
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
        [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06], dist)
    db.insert("Combinatorics: permutations combinations generating functions", "math",
        [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07], dist)
    # Food
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07], dist)
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
        [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08], dist)
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06], dist)
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09], dist)
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
        [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06], dist)
    # Sports
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72], dist)
    db.insert("Football: tackles touchdowns field goals and strategy", "sports",
        [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76], dist)
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82], dist)
    db.insert("Chess: openings endgames tactics strategic board game", "sports",
        [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90], dist)
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80], dist)

class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

class VectorDBRequestHandler(http.server.BaseHTTPRequestHandler):

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # Serve index.html
        if path in ("/", "/index.html"):
            try:
                with open("index.html", "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
            return

        # /status
        if path == "/status":
            available = ollama.is_available()
            res = {
                "ollamaAvailable": available,
                "embedModel": ollama.embedModel,
                "genModel": ollama.genModel,
                "docCount": doc_db.size(),
                "docDims": doc_db.dims,
                "demoDims": DIMS,
                "demoCount": db.size()
            }
            self.send_json_response(res)
            return

        # /stats
        if path == "/stats":
            res = {
                "count": db.size(),
                "dims": DIMS,
                "algorithms": ["bruteforce", "kdtree", "hnsw"],
                "metrics": ["euclidean", "cosine", "manhattan"]
            }
            self.send_json_response(res)
            return

        # /items
        if path == "/items":
            items = db.all_items()
            res = []
            for item in items:
                res.append({
                    "id": item["id"],
                    "metadata": item["metadata"],
                    "category": item["category"],
                    "embedding": item["emb"]
                })
            self.send_json_response(res)
            return

        # /search
        if path == "/search":
            v_str = query.get("v", [""])[0]
            k_str = query.get("k", ["5"])[0]
            metric = query.get("metric", ["cosine"])[0]
            algo = query.get("algo", ["hnsw"])[0]

            try:
                q = [float(x) for x in v_str.split(",") if x.strip()]
            except ValueError:
                self.send_error_response("invalid vector query")
                return

            if len(q) != DIMS:
                self.send_error_response(f"need {DIMS}D vector")
                return

            try:
                k = int(k_str)
            except ValueError:
                k = 5

            hits, us, algo_used, metric_used = db.search(q, k, metric, algo)
            
            res_hits = []
            for h in hits:
                res_hits.append({
                    "id": h["id"],
                    "metadata": h["metadata"],
                    "category": h["category"],
                    "distance": h["distance"],
                    "embedding": h["emb"]
                })
            
            res = {
                "results": res_hits,
                "latencyUs": us,
                "algo": algo_used,
                "metric": metric_used
            }
            self.send_json_response(res)
            return

        # /benchmark
        if path == "/benchmark":
            v_str = query.get("v", [""])[0]
            k_str = query.get("k", ["5"])[0]
            metric = query.get("metric", ["cosine"])[0]

            try:
                q = [float(x) for x in v_str.split(",") if x.strip()]
            except ValueError:
                self.send_error_response("invalid vector query")
                return

            if len(q) != DIMS:
                self.send_error_response(f"need {DIMS}D vector")
                return

            try:
                k = int(k_str)
            except ValueError:
                k = 5

            bf_us, kd_us, hnsw_us, n = db.benchmark(q, k, metric)
            res = {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "itemCount": n
            }
            self.send_json_response(res)
            return

        # /hnsw-info
        if path == "/hnsw-info":
            info = db.hnsw_info()
            self.send_json_response(info)
            return

        # /doc/list
        if path == "/doc/list":
            docs = doc_db.all_items()
            res = []
            for d in docs:
                preview = d["text"][:120]
                if len(d["text"]) > 120:
                    preview += "…"
                words = len(d["text"].split())
                res.append({
                    "id": d["id"],
                    "title": d["title"],
                    "preview": preview,
                    "words": words
                })
            self.send_json_response(res)
            return

        # Path not found
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)
        body_str = body_bytes.decode("utf-8")
        
        try:
            data = json.loads(body_str) if body_str else {}
        except json.JSONDecodeError:
            self.send_error_response("invalid JSON body")
            return

        # /insert
        if path == "/insert":
            meta = data.get("metadata", "")
            cat = data.get("category", "")
            emb = data.get("embedding", [])

            if not meta or not emb or len(emb) != DIMS:
                self.send_error_response("invalid body")
                return

            try:
                emb = [float(x) for x in emb]
            except ValueError:
                self.send_error_response("invalid embedding floats")
                return

            item_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
            self.send_json_response({"id": item_id})
            return

        # /doc/insert
        if path == "/doc/insert":
            title = data.get("title", "")
            text = data.get("text", "")

            if not title or not text:
                self.send_error_response("need title and text")
                return

            chunks = chunk_text(text, 250, 30)
            ids = []

            for i, chunk in enumerate(chunks):
                emb = ollama.embed(chunk)
                if not emb:
                    self.send_error_response(
                        "Ollama unavailable. Install from https://ollama.com then run: "
                        "ollama pull nomic-embed-text && ollama pull llama3.2"
                    )
                    return
                
                chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
                ids.append(doc_db.insert(chunk_title, chunk, emb))

            self.send_json_response({
                "ids": ids,
                "chunks": len(chunks),
                "dims": doc_db.dims
            })
            return

        # /doc/search
        if path == "/doc/search":
            question = data.get("question", "")
            k = int(data.get("k", 3))

            if not question:
                self.send_error_response("need question")
                return

            q_emb = ollama.embed(question)
            if not q_emb:
                self.send_error_response("Ollama unavailable")
                return

            hits = doc_db.search(q_emb, k)
            contexts = []
            for dist, doc in hits:
                contexts.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "distance": dist
                })
            self.send_json_response({"contexts": contexts})
            return

        # /doc/ask
        if path == "/doc/ask":
            question = data.get("question", "")
            k = int(data.get("k", 3))

            if not question:
                self.send_error_response("need question")
                return

            q_emb = ollama.embed(question)
            if not q_emb:
                self.send_error_response("Ollama unavailable")
                return

            hits = doc_db.search(q_emb, k)
            
            ctx_parts = []
            for i, (dist, doc) in enumerate(hits):
                ctx_parts.append(f"[{i+1}] {doc['title']}:\n{doc['text']}\n\n")
            ctx_str = "".join(ctx_parts)

            prompt = (
                "You are a helpful assistant. Answer the user's question directly. "
                "Use the provided context if it contains relevant information. "
                "If it doesn't, just use your own general knowledge. "
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                "Just answer the question naturally.\n\n"
                f"Context:\n{ctx_str}"
                f"Question: {question}\n\n"
                "Answer:"
            )

            answer = ollama.generate(prompt)

            contexts_res = []
            for dist, doc in hits:
                contexts_res.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "distance": dist
                })

            self.send_json_response({
                "answer": answer,
                "model": ollama.genModel,
                "contexts": contexts_res,
                "docCount": doc_db.size()
            })
            return

        self.send_response(404)
        self.end_headers()

    def do_DELETE(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Match /delete/(\d+)
        delete_match = re.match(r"^/delete/(\d+)$", path)
        if delete_match:
            item_id = int(delete_match.group(1))
            ok = db.remove(item_id)
            self.send_json_response({"ok": ok})
            return

        # Match /doc/delete/(\d+)
        doc_delete_match = re.match(r"^/doc/delete/(\d+)$", path)
        if doc_delete_match:
            item_id = int(doc_delete_match.group(1))
            ok = doc_db.remove(item_id)
            self.send_json_response({"ok": ok})
            return

        self.send_response(404)
        self.end_headers()

    def send_json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_error_response(self, err_msg):
        self.send_response(200)  # matching original C++ API returning 200 with error JSON
        self.send_header("Content-Type", "application/json")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"error": err_msg}).encode("utf-8"))

# =====================================================================
#  MAIN ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    load_demo_data()
    
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embedModel}  gen model: {ollama.genModel}")

    port = int(os.environ.get("PORT", 8080))
    server_address = ("", port)
    try:
        with ThreadedHTTPServer(server_address, VectorDBRequestHandler) as httpd:
            print(f"Python server listening on port {port}...")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
