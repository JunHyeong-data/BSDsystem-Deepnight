"""한 프레임 상세 진단 — mask instance ID vs GT projection 매칭.

실행: python3 scripts/bsd_debug.py
출력: /tmp/bsd_debug_rgb.jpg (시각화), /tmp/bsd_debug_mask.jpg
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
from collect_data import world_to_pixel, extract_bboxes, MIN_AREA

rclpy.init()
n = Node("dbg")
buf = {}
qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                 history=QoSHistoryPolicy.KEEP_LAST, depth=1)
n.create_subscription(CompressedImage, "/image_jpeg/rgb",
                      lambda m: buf.update(rgb=cv2.imdecode(np.frombuffer(m.data, np.uint8), 1)), qos)
n.create_subscription(CompressedImage, "/image_jpeg/compressed",
                      lambda m: buf.update(mask=cv2.imdecode(np.frombuffer(m.data, np.uint8), 1)), qos)
n.create_subscription(EgoVehicleStatus, "/Ego_topic",
                      lambda m: buf.update(ego=m), qos)
n.create_subscription(ObjectStatusList, "/Object_topic",
                      lambda m: buf.update(obj=m), qos)

t0 = time.time()
needed = {"rgb", "mask", "ego", "obj"}
while time.time()-t0 < 12 and not needed.issubset(buf.keys()):
    rclpy.spin_once(n, timeout_sec=0.1)
missing = needed - buf.keys()
if missing:
    print(f"ERR: missing topics after 12s: {missing}")
    sys.exit(1)
ego, obj, mask, rgb = buf["ego"], buf["obj"], buf["mask"], buf["rgb"]

# === Mask 전체 unique BGR 분석 ===
flat = mask.reshape(-1, 3)
uniq, counts = np.unique(flat, axis=0, return_counts=True)
order = np.argsort(-counts)
print(f"=== MASK unique BGR (top 12 by pixel count) ===")
for k in order[:12]:
    print(f"   BGR {tuple(int(x) for x in uniq[k])}  count={counts[k]:6d}")

# === 각 bbox 의 dominant non-white color ===
bboxes = extract_bboxes(mask)
print(f"\n=== BBOX (count={len(bboxes)}) ===")
bbox_color = {}
for i, (x1,y1,x2,y2) in enumerate(bboxes):
    roi = mask[y1:y2, x1:x2].reshape(-1, 3)
    not_white = ~np.all(roi >= 240, axis=1)
    fg = roi[not_white]
    if len(fg) == 0:
        print(f"  [{i}] ({x1},{y1})-({x2},{y2})  size={x2-x1}x{y2-y1}  area=0 (all white?)")
        bbox_color[i] = None
        continue
    fg_uniq, fg_counts = np.unique(fg, axis=0, return_counts=True)
    dom = fg_uniq[np.argmax(fg_counts)]
    print(f"  [{i}] ({x1},{y1})-({x2},{y2})  size={x2-x1}x{y2-y1}  fg_pixels={len(fg)}  dominant BGR={tuple(int(x) for x in dom)}")
    bbox_color[i] = tuple(int(x) for x in dom)

# === GT projection ===
print(f"\n=== GT projection check ===")
all_objs = [("PED", o) for o in obj.pedestrian_list] + [("NPC", o) for o in obj.npc_list]
for tag, o in all_objs:
    px = world_to_pixel(o.position, ego.position, ego.heading)
    dx = o.position.x - ego.position.x
    dy = o.position.y - ego.position.y
    h = np.deg2rad(ego.heading)
    vX = dx*np.sin(h) + dy*np.cos(h)
    vY = dx*np.cos(h) - dy*np.sin(h)
    print(f"{tag} '{o.name}'  veh_frame X(fwd)={vX:+.2f}  Y(right)={vY:+.2f}  -> px {px}")
    if px is not None and 0 <= px[0] < mask.shape[1] and 0 <= px[1] < mask.shape[0]:
        mc = tuple(int(x) for x in mask[px[1], px[0]])
        nbhd = mask[max(0,px[1]-1):px[1]+2, max(0,px[0]-1):px[0]+2].reshape(-1, 3)
        ncols = [tuple(int(x) for x in c) for c in np.unique(nbhd, axis=0)]
        print(f"    mask color at proj: BGR {mc}  3x3 colors: {ncols[:5]}")
        hits = []
        for i, (x1,y1,x2,y2) in enumerate(bboxes):
            if x1 <= px[0] <= x2 and y1 <= px[1] <= y2:
                hits.append(i)
        print(f"    falls inside bbox idx: {hits}")
        for i in hits:
            print(f"        bbox[{i}] dominant={bbox_color.get(i)}  proj_color_matches_dom={mc == bbox_color.get(i)}")

# === 시각화 ===
dbg = rgb.copy()
for i, (x1,y1,x2,y2) in enumerate(bboxes):
    cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(dbg, f"#{i}", (x1+2, y1+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
for tag, o in all_objs:
    px = world_to_pixel(o.position, ego.position, ego.heading)
    if px is not None:
        col = (0,0,255) if tag == "NPC" else (255,0,255)
        cv2.drawMarker(dbg, px, col, cv2.MARKER_CROSS, 12, 2)
        cv2.putText(dbg, f"{tag} {o.name[:10]}", (px[0]+8, px[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
cv2.imwrite("/tmp/bsd_debug_rgb.jpg", dbg)
cv2.imwrite("/tmp/bsd_debug_mask.jpg", mask)
print(f"\nSaved /tmp/bsd_debug_rgb.jpg, /tmp/bsd_debug_mask.jpg")
n.destroy_node(); rclpy.shutdown()
