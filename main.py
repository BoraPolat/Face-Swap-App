"""
FaceSwap Backend Module
========================
The main engine performing face swap operations.
"""

import cv2, os, math
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import mediapipe as mp
from scipy.spatial import Delaunay

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "inswapper_128.onnx"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def robust_read_image(path):
    try:
        with open(path, "rb") as f:
            img = cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None: return None
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    except Exception as e:
        print(f"Image reading error: {e}")
        return None

# ============================================================================
# DEEP LEARNING ENGINE (InsightFace)
# ============================================================================

class DeepLearningEngine:
    def __init__(self, device='gpu'):
        print(f"🚀 [AI] DL Engine Initializing ({device})...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'gpu' in device else ['CPUExecutionProvider']
        self.face_analyser = FaceAnalysis(name='buffalo_l', providers=providers)
        self.face_analyser.prepare(ctx_id=0, det_size=(960, 960))
        self.swapper = insightface.model_zoo.get_model(MODEL_PATH, download=False, download_zip=False, providers=providers)
        self.face_cache = {}

    def get_faces(self, img_path):
        if img_path in self.face_cache: 
            return self.face_cache[img_path]
        img = robust_read_image(img_path)
        if img is None: return []
        faces = self.face_analyser.get(img)
        sorted_faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        self.face_cache[img_path] = sorted_faces
        return sorted_faces

    def process(self, img_a, img_b, idx_a=0, idx_b=0, show_debug=False):
        faces_a = sorted(self.face_analyser.get(img_a), 
                        key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        if not faces_a: return img_b
        face_a = faces_a[idx_a] if idx_a < len(faces_a) else faces_a[0]

        faces_b = sorted(self.face_analyser.get(img_b), 
                        key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        if not faces_b: return img_b
        face_b = faces_b[idx_b] if idx_b < len(faces_b) else faces_b[0]

        res = self.swapper.get(img_b, face_b, face_a, paste_back=True)
        
        if show_debug:
            res = res.copy()
            if face_b.kps is not None:
                for p in face_b.kps:
                    pt = (int(p[0]), int(p[1]))
                    cv2.circle(res, pt, 3, (0, 0, 0), -1)
                    cv2.circle(res, pt, 2, (0, 255, 255), -1)
            if face_b.bbox is not None:
                bbox = face_b.bbox.astype(int)
                cv2.rectangle(res, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        return self._enhance(res, face_b)

    def _enhance(self, swapped_img, target_face):
        try:
            bbox = target_face.bbox.astype(int)
            h, w = swapped_img.shape[:2]
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w, bbox[2]), min(h, bbox[3])
            face_region = swapped_img[y1:y2, x1:x2]
            kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5.0, -0.5], [-0.5, -0.5, -0.5]])
            sharpened = cv2.filter2D(face_region, -1, kernel)
            swapped_img[y1:y2, x1:x2] = cv2.addWeighted(face_region, 0.7, sharpened, 0.3, 0)
        except: pass
        return swapped_img

# ============================================================================
# GEOMETRIC ENGINE (MediaPipe + Delaunay)
# ============================================================================

class GeometricEngine:
    def __init__(self):
        print("📐 [GEO] Preparing Geometric Engine...")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3)

    def get_landmarks(self, img, face_idx=0): 
        if img is None: return None
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) <= face_idx:
            return None

        h, w = img.shape[:2]
        return np.array([(int(l.x * w), int(l.y * h))
                        for l in results.multi_face_landmarks[face_idx].landmark], dtype=np.int32)

    def process(self, img_a, img_b, idx_a=0, idx_b=0, show_debug=False):
        # FIX: get_landmarks now takes idx_a and idx_b parameters
        lm_a = self.get_landmarks(img_a, idx_a)
        lm_b = self.get_landmarks(img_b, idx_b)
        
        # FIX: Replaced array check with 'is None' which was the source of the error
        if lm_a is None or lm_b is None: 
            return img_b, lm_a, lm_b

        warped_face = np.zeros_like(img_b)
        debug_triangles = []

        try:
            delaunay = Delaunay(lm_b)
            for indices in delaunay.simplices:
                t1, t2 = lm_a[indices], lm_b[indices]
                if show_debug: debug_triangles.append(t2)
                
                r1, r2 = cv2.boundingRect(t1), cv2.boundingRect(t2)
                t1_rect, t2_rect = t1 - [r1[0], r1[1]], t2 - [r2[0], r2[1]]
                
                img1_rect = img_a[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
                M = cv2.getAffineTransform(np.float32(t1_rect[:3]), np.float32(t2_rect[:3]))
                dst = cv2.warpAffine(img1_rect, M, (r2[2], r2[3]), borderMode=cv2.BORDER_REFLECT_101)
                
                mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
                cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0))
                
                warped_slice = warped_face[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
                warped_face[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = warped_slice * (1 - mask) + dst * mask

            hull = cv2.convexHull(lm_b)
            mask_clone = np.zeros_like(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY))
            cv2.fillConvexPoly(mask_clone, hull, 255)
            
            center = cv2.moments(hull)
            center_pt = (int(center["m10"]/center["m00"]), int(center["m01"]/center["m00"]))
            result_img = cv2.seamlessClone(warped_face, img_b, mask_clone, center_pt, cv2.NORMAL_CLONE)

            if show_debug:
                for tri in debug_triangles:
                    for i in range(3):
                        cv2.line(result_img, tuple(tri[i]), tuple(tri[(i+1)%3]), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(result_img, [hull], True, (255, 0, 0), 2)
            
            return result_img, lm_a, lm_b
            
        except Exception as e:
            print(f"Geo Error: {e}")
            return img_b, lm_a, lm_b

# ============================================================================
# PUBLIC API
# ============================================================================

_dl_engine = _geo_engine = None

def get_faces_info(image_path, device='gpu'):
    global _dl_engine
    if _dl_engine is None: _dl_engine = DeepLearningEngine(device)
    faces = _dl_engine.get_faces(image_path)
    return [{'index': i, 'confidence': float(f.det_score), 'bbox': f.bbox.tolist(), 
             'size': int((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))} for i, f in enumerate(faces)]

def process_comparison(path_a, path_b, mode="Deep Learning", idx_a=0, idx_b=0, 
                      device='gpu', show_debug=False, progress_callback=None):
    global _dl_engine, _geo_engine
    
    # STEP 1: Engine Preparation (5% - 15%)
    if progress_callback: progress_callback(5, "Checking engine...")
    
    if "Deep" in mode:
        if _dl_engine is None: 
            # Model loading takes time on first run
            if progress_callback: progress_callback(10, "Loading AI model (This may take a while)...")
            _dl_engine = DeepLearningEngine(device)
        engine = _dl_engine
    else:
        if _geo_engine is None: 
            if progress_callback: progress_callback(10, "Preparing geometric engine...")
            _geo_engine = GeometricEngine()
        engine = _geo_engine

    # STEP 2: Reading Images (20%)
    if progress_callback: progress_callback(20, "Processing images...")
    img_a, img_b = robust_read_image(path_a), robust_read_image(path_b)
    if img_a is None or img_b is None: raise Exception("Image could not be loaded!")

    try:
        # STEP 3: Processing (30% - 80%)
        if "Deep" in mode:
            # First face swap
            if progress_callback: progress_callback(30, "1. Swapping face (A -> B)...")
            res1 = engine.process(img_a, img_b, idx_a, idx_b, show_debug)
            
            # Second face swap
            if progress_callback: progress_callback(60, "2. Swapping face (B -> A)...")
            res2 = engine.process(img_b, img_a, idx_b, idx_a, show_debug)
            
            # Success rate calculation
            if progress_callback: progress_callback(85, "Analyzing results...")
            faces_a, faces_b = _dl_engine.get_faces(path_a), _dl_engine.get_faces(path_b)
            success_rate = calculate_success_rate(faces_a, faces_b, idx_a, idx_b)
            
        else:
            # Geometric mode operations
            if progress_callback: progress_callback(30, "Geometric calculation 1/2...")
            res1, lm_a1, lm_b1 = engine.process(img_a, img_b, idx_a, idx_b, show_debug)
            
            if progress_callback: progress_callback(60, "Geometric calculation 2/2...")
            res2, lm_b2, lm_a2 = engine.process(img_b, img_a, idx_b, idx_a, show_debug)
            
            if progress_callback: progress_callback(85, "Calculating accuracy rate...")
            rate1 = calculate_success_rate_geometric(lm_a1, lm_b1)
            rate2 = calculate_success_rate_geometric(lm_b2, lm_a2)
            success_rate = (rate1 + rate2) / 2.0

        # STEP 4: Completion (100%)
        if progress_callback: progress_callback(100, "Process Completed!")
        return res1, res2, success_rate
        
    except Exception as e:
        print(f"Error: {e}")
        # Return original images in case of error
        return img_b, img_a, 0.0
        
def get_available_devices():
    """
    Checks for available devices in the system in real-time.
    Returns:
        list: List of available devices [{'id': str, 'name': str, 'available': bool}, ...]
    """
    devices = []
    
    # CPU is always available
    devices.append({
        'id': 'cpu', 
        'name': 'CPU (Standard)', 
        'available': True,
        'info': 'Always available'
    })
    
    # GPU check
    gpu_available = False
    gpu_name = "GPU"
    gpu_info = ""
    
    try:
        import onnxruntime as ort
        
        # CUDA provider check
        available_providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in available_providers:
            gpu_available = True
            
            # Get GPU info
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    gpu_name = result.stdout.strip().split('\n')[0]
                    gpu_info = f"NVIDIA CUDA available"
            except:
                gpu_name = "NVIDIA GPU"
                gpu_info = "CUDA support available"
        else:
            gpu_info = "CUDA provider not found"
            
    except ImportError:
        gpu_info = "onnxruntime not installed"
    
    devices.append({
        'id': 'gpu',
        'name': gpu_name,
        'available': gpu_available,
        'info': gpu_info
    })
    
    return devices

def calculate_success_rate(faces_a, faces_b, idx_a=0, idx_b=0):
    try:
        face_a = faces_a[idx_a] if idx_a < len(faces_a) else faces_a[0]
        face_b = faces_b[idx_b] if idx_b < len(faces_b) else faces_b[0]
        return round((face_a.det_score + face_b.det_score) / 2 * 100, 1)
    except: return 75.0

def calculate_success_rate_geometric(lm_a, lm_b, tolerance=0.50):
    try:
        # FIX: Success rate calculation check also updated to use 'is None'
        if lm_a is None or lm_b is None: return 0.0
        dists = np.linalg.norm(lm_a.astype(np.float32) - lm_b.astype(np.float32), axis=1)
        diag = math.hypot(max(lm_a[:,0].max(), lm_b[:,0].max()) - min(lm_a[:,0].min(), lm_b[:,0].min()), 
                         max(lm_a[:,1].max(), lm_b[:,1].max()) - min(lm_a[:,1].min(), lm_b[:,1].min()))
        if diag <= 0: return 0.0
        score = (1.0 - (np.mean(dists) / diag / tolerance)) * 100.0
        return round(max(0.0, min(100.0, score)), 2)
    except: return 0.0