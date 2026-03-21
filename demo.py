import numpy as np
from ultralytics import YOLO
from owslib.wmts import WebMapTileService
from PIL import Image
import io
import cv2
import folium
import math
from pyproj import Transformer
import requests
import json
import gradio as gr


# ─────────────────────────────────────────────────────────────
# Satellite image retrieval
# ─────────────────────────────────────────────────────────────

def wmts_tile_to_array(tile_data):
    img = Image.open(io.BytesIO(tile_data.read()))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def get_tile_indices(xmin, ymin, xmax, ymax, matrix):
    tile_size = matrix.tilewidth * matrix.scaledenominator * 0.28e-3
    origin_x, origin_y = matrix.topleftcorner
    col_min = int((xmin - origin_x) // tile_size)
    col_max = int((xmax - origin_x) // tile_size)
    row_min = int((origin_y - ymax) // tile_size)
    row_max = int((origin_y - ymin) // tile_size)
    return col_min, col_max, row_min, row_max


def retrieve_satelite_image(top_left_corner, bottom_right_corner, progress_cb=None):
    wmts_url = (
        "https://cartografia.dgterritorio.gov.pt/ortos2018/service"
        "?service=WMTS&request=GetCapabilities"
    )
    wmts = WebMapTileService(wmts_url)

    xmin, ymax = top_left_corner
    xmax, ymin = bottom_right_corner

    layer          = "Ortos2018-RGB"
    tile_matrix_set = "PTTM_06"
    zoom_level     = "14"
    matrix   = wmts.tilematrixsets[tile_matrix_set].tilematrix[zoom_level]
    res      = matrix.scaledenominator * 0.28e-3
    tile_size_m = matrix.tilewidth * res

    col_min, col_max, row_min, row_max = get_tile_indices(xmin, ymin, xmax, ymax, matrix)
    n_rows = row_max + 1 - row_min
    n_cols = col_max + 1 - col_min
    total_tiles = n_rows * n_cols
    processed_blocks = np.empty((n_rows, n_cols), dtype=object)

    done = 0
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            tile = wmts.gettile(
                layer=layer, tilematrixset=tile_matrix_set,
                tilematrix=zoom_level, row=row, column=col, format="image/png",
            )
            processed_blocks[row - row_min, col - col_min] = {"img": wmts_tile_to_array(tile)}
            done += 1
            if progress_cb:
                progress_cb(done, total_tiles)

    block_h, block_w = processed_blocks[0, 0]["img"].shape[:2]
    stitched = np.zeros((n_rows * block_h, n_cols * block_w, 3), dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            img = processed_blocks[row, col]["img"]
            stitched[row*block_h:(row+1)*block_h, col*block_w:(col+1)*block_w] = img

    origin_x = matrix.topleftcorner[0] + col_min * tile_size_m
    origin_y = matrix.topleftcorner[1] - row_min * tile_size_m

    col_start = int(round((xmin - origin_x) / res))
    row_start = int(round((origin_y - ymax) / res))
    col_end   = int(round((xmax - origin_x) / res))
    row_end   = int(round((origin_y - ymin) / res))
    satellite_image = stitched[row_start:row_end, col_start:col_end, :]

    def conversion(x, y):
        return int((x - xmin) / res), int((ymax - y) / res)

    return satellite_image, res, conversion


# ─────────────────────────────────────────────────────────────
# Model & data structures
# ─────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CachedModel:
    def __init__(self):
        self.model = YOLO("best.pt")
        self.model.to("cpu")
        self.model.eval()


class RoofPlane:
    def __init__(self, face, probability, corners, normal, inclination, orientation):
        self.face = face;  self.probability = probability
        self.corners = corners;  self.normal = normal
        self.inclination = inclination;  self.orientation = orientation


class EstimatedBuilding:
    def __init__(self):
        self.box_coords_in_epsg_3763: list[float] = []
        self.planes_in_physical_dimensions: list[RoofPlane] = []

    def convert_tensor_prediction_to_building(
        self, boxes_xywhn, roof_prediction, top_left_corner, res, image_shape
    ):
        img_h, img_w = image_shape[:2]
        xmin_map, ymax_map = top_left_corner
        cx_norm, cy_norm, w_norm, h_norm = boxes_xywhn
        cx_map = xmin_map + cx_norm * img_w * res
        cy_map = ymax_map - cy_norm * img_h * res
        w_map  = w_norm * img_w * res
        h_map  = h_norm * img_h * res

        self.box_coords_in_epsg_3763 = [
            cx_map - w_map/2, cy_map - h_map/2,
            cx_map + w_map/2, cy_map + h_map/2,
        ]

        probh, probtop, probright, probbottom, probleft = roof_prediction[0:5]
        inc_top, inc_right, inc_bottom, inc_left = roof_prediction[10:14]
        ori_top, ori_right, ori_bottom, ori_left = roof_prediction[14:18]

        dx, dy = w_map/2, h_map/2
        p_tl = np.array([cx_map-dx, cy_map+dy, 0.])
        p_tr = np.array([cx_map+dx, cy_map+dy, 0.])
        p_br = np.array([cx_map+dx, cy_map-dy, 0.])
        p_bl = np.array([cx_map-dx, cy_map-dy, 0.])

        for face_name, prob, inc, ori, base_az, ps, pe in [
            ("top",    probtop,    inc_top,    ori_top,     math.pi/2,  p_tl, p_tr),
            ("right",  probright,  inc_right,  ori_right,   0.,         p_tr, p_br),
            ("bottom", probbottom, inc_bottom, ori_bottom, -math.pi/2,  p_br, p_bl),
            ("left",   probleft,   inc_left,   ori_left,    math.pi,    p_bl, p_tl),
        ]:
            if prob < 0.5: continue
            tilt    = inc * (math.pi/2)
            azimuth = (ori - 0.5) * (math.pi/2) + base_az
            normal  = np.array([math.sin(tilt)*math.cos(azimuth),
                                 math.sin(tilt)*math.sin(azimuth),
                                 math.cos(tilt)])
            normal /= np.linalg.norm(normal)
            edge_unit = (pe - ps) / np.linalg.norm(pe - ps)
            slope_vec = np.cross(edge_unit, normal)
            if slope_vec[2] > 0: slope_vec = -slope_vec
            sl = max(w_map, h_map) * 0.2
            self.planes_in_physical_dimensions.append(RoofPlane(
                face=face_name, probability=float(prob),
                corners=[ps, pe, pe+slope_vec*sl, ps+slope_vec*sl],
                normal=normal, inclination=float(inc), orientation=float(ori),
            ))


# ─────────────────────────────────────────────────────────────
# Prediction / OSM / map
# ─────────────────────────────────────────────────────────────

def retrieve_prediction_list(satellite_image, top_left_corner, res,
                              building_threshold, overlap_threshold, cached_model):
    results = cached_model.model(satellite_image, conf=building_threshold,
                                  iou=overlap_threshold)[0]
    predictions = []
    for i in range(results.boxes.shape[0]):
        b = EstimatedBuilding()
        b.convert_tensor_prediction_to_building(
            boxes_xywhn=np.array(results.boxes[i].xywhn[0]),
            roof_prediction=np.array(sigmoid(results.roof.data[i, :])),
            top_left_corner=top_left_corner, res=res,
            image_shape=satellite_image.shape,
        )
        predictions.append(b)
    return predictions


def get_osm_buildings(top_left_corner, bottom_right_corner):
    tr = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)
    tl_lon, tl_lat = tr.transform(*top_left_corner)
    br_lon, br_lat = tr.transform(*bottom_right_corner)
    s, n = min(tl_lat, br_lat), max(tl_lat, br_lat)
    w, e = min(tl_lon, br_lon), max(tl_lon, br_lon)
    q = f"""[out:json];(way["building"]({s},{w},{n},{e});
    relation["building"]({s},{w},{n},{e}););out body;>;out skel qt;"""
    r = requests.post("https://overpass-api.de/api/interpreter", data=q)
    r.raise_for_status()
    data  = r.json()
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"]=="node"}
    features = []
    for el in data["elements"]:
        if el["type"] != "way": continue
        coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
        if len(coords) < 3: continue
        features.append({"type":"Feature",
                          "properties":{"osm_id":el["id"], **el.get("tags",{})},
                          "geometry":{"type":"Polygon","coordinates":[coords]}})
    return {"type":"FeatureCollection","features":features}


def build_map_html(osm_geojson, predicted_buildings, satellite_image,
                   top_left_corner, bottom_right_corner):
    tr = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)

    def box_poly(box):
        x0,y0,x1,y1 = box
        return [tr.transform(x,y)[::-1] for x,y in [(x0,y1),(x1,y1),(x1,y0),(x0,y0),(x0,y1)]]

    lats, lons = [], []
    for f in osm_geojson["features"]:
        for lon, lat in f["geometry"]["coordinates"][0]:
            lats.append(lat); lons.append(lon)
    center = [sum(lats)/len(lats), sum(lons)/len(lons)]

    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap")

    tl_lon, tl_lat = tr.transform(*top_left_corner)
    br_lon, br_lat = tr.transform(*bottom_right_corner)
    s,n = min(tl_lat,br_lat), max(tl_lat,br_lat)
    w,e = min(tl_lon,br_lon), max(tl_lon,br_lon)

    sat = folium.FeatureGroup(name="Satellite Image")
    folium.raster_layers.ImageOverlay(image=satellite_image,
        bounds=[[s,w],[n,e]], opacity=0.85,
        interactive=False, cross_origin=False).add_to(sat)
    sat.add_to(m)

    osm = folium.FeatureGroup(name="OSM Buildings")
    folium.GeoJson(osm_geojson,
        style_function=lambda _: {"fillColor":"#3388ff","color":"#1a55cc",
                                   "weight":2,"fillOpacity":0.3},
        tooltip=folium.GeoJsonTooltip(fields=["osm_id","building"],
                                       aliases=["OSM ID","Type"], localize=True)
    ).add_to(osm)
    osm.add_to(m)

    pred = folium.FeatureGroup(name="Model Predictions")
    for i, building in enumerate(predicted_buildings):
        if not building.box_coords_in_epsg_3763: continue
        n_planes = len(building.planes_in_physical_dimensions)
        folium.Polygon(locations=box_poly(building.box_coords_in_epsg_3763),
            color="#cc0000", fill_color="#ff4444", fill_opacity=0.3, weight=2,
            tooltip=f"Building {i} — {n_planes} roof plane(s)").add_to(pred)
        for plane in building.planes_in_physical_dimensions:
            cx = sum(p[0] for p in plane.corners)/4
            cy = sum(p[1] for p in plane.corners)/4
            lon, lat = tr.transform(cx, cy)
            folium.CircleMarker(location=[lat,lon], radius=4,
                color="#ff9900", fill=True, fill_opacity=0.8,
                tooltip=(f"Face: {plane.face}<br>Prob: {plane.probability:.2f}<br>"
                         f"Inc: {plane.inclination:.2f}<br>Ori: {plane.orientation:.2f}")
            ).add_to(pred)
    pred.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


# ─────────────────────────────────────────────────────────────
# One-time model load + tile cache
# ─────────────────────────────────────────────────────────────

model = CachedModel()

# Keyed by (west, south, east, north) rounded to 7 decimal places.
# Stores (satellite_image, res) so repeated runs on the same region
# skip the WMTS download entirely.
_tile_cache: dict = {}

def _bbox_key(west, south, east, north):
    return tuple(round(v, 7) for v in (west, south, east, north))


# ─────────────────────────────────────────────────────────────
# Selection map — self-contained srcdoc iframe
# The iframe owns its own document so Leaflet initialises cleanly.
# When the user draws/edits a rectangle it sends a postMessage to
# the parent Gradio page, which a tiny <script> in the outer page
# catches and writes into the hidden bbox-bridge textbox.
# ─────────────────────────────────────────────────────────────

_IFRAME_SRCDOC = """
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1'/>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
<link rel='stylesheet' href='https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css'/>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body, #map { width:100%; height:100%; }
</style>
</head>
<body>
<div id='map'></div>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
<script src='https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js'></script>
<script>
  var map = L.map('map').setView([38.7167, -9.1333], 14);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19
  }).addTo(map);

  var drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  var drawControl = new L.Control.Draw({
    draw: {
      rectangle: { shapeOptions: { color:'#3b82f6', weight:2, fillOpacity:0.12 } },
      polygon: false, circle: false, marker: false,
      circlemarker: false, polyline: false
    },
    edit: { featureGroup: drawnItems }
  });
  map.addControl(drawControl);

  function sendBbox(bounds) {
    window.parent.postMessage({
      type: 'bbox',
      data: {
        north: bounds.getNorth(), south: bounds.getSouth(),
        east:  bounds.getEast(),  west:  bounds.getWest()
      }
    }, '*');
  }

  map.on(L.Draw.Event.CREATED, function(e) {
    drawnItems.clearLayers();
    drawnItems.addLayer(e.layer);
    sendBbox(e.layer.getBounds());
  });
  map.on(L.Draw.Event.EDITED, function(e) {
    e.layers.eachLayer(function(l) { sendBbox(l.getBounds()); });
  });
  map.on(L.Draw.Event.DELETED, function() {
    window.parent.postMessage({ type: 'bbox', data: null }, '*');
  });
</script>
</body>
</html>
""".replace('"', '&quot;').replace("'", "&#39;")  # escape for srcdoc attribute

SELECTION_MAP_HTML = f"""
<iframe
  srcdoc="{_IFRAME_SRCDOC}"
  style="width:100%; height:460px; border:1px solid #1a2540;
         border-radius:10px; display:block;"
  sandbox="allow-scripts allow-same-origin"
  allowfullscreen>
</iframe>
"""

# Injected after the full Gradio DOM is ready via demo.load(js=...)
_BBOX_LISTENER_JS = """
() => {
    if (window._bboxListenerAdded) return;
    window._bboxListenerAdded = true;
    window.addEventListener('message', function(ev) {
        if (!ev.data || ev.data.type !== 'bbox') return;
        var el = document.querySelector('#bbox-bridge textarea');
        if (!el) return;
        el.value = ev.data.data ? JSON.stringify(ev.data.data) : '';
        el.dispatchEvent(new Event('input', { bubbles: true }));
    });
}
"""

_EMPTY_MAP = """
<div style="
    height:560px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    background:#060c1a; border-radius:12px;
    border:1px solid #1a2540; gap:14px;
">
  <svg width='52' height='52' viewBox='0 0 24 24' fill='none'
       stroke='#1e3a5f' stroke-width='1.2' stroke-linecap='round'>
    <rect x='2' y='2' width='20' height='20' rx='3'/>
    <path d='M2 9h20M9 21V9'/>
  </svg>
  <span style='color:#334155; font-family:monospace; font-size:12px;
               letter-spacing:0.05em; text-align:center; line-height:1.8;'>
    ① Draw a rectangle on the map<br>② Set thresholds<br>③ Click Run Detection
  </span>
</div>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

body, .gradio-container { background:#060c1a !important; }
.gradio-container {
    max-width:1500px !important;
    font-family:'JetBrains Mono', monospace !important;
}
#hdr { padding:28px 0 18px; border-bottom:1px solid #1a2540; margin-bottom:18px; }
#hdr h1 { font-size:20px !important; font-weight:700 !important;
           color:#e2e8f0 !important; letter-spacing:-0.02em; margin:0 !important; }
#hdr p  { color:#475569 !important; font-size:11px !important;
           margin:4px 0 0 !important; text-transform:uppercase; letter-spacing:0.08em; }
#left-panel { background:#0c1525; border:1px solid #1a2540;
               border-radius:12px; padding:20px; }
.slabel { color:#475569; font-size:10px; text-transform:uppercase;
          letter-spacing:0.1em; margin:18px 0 8px;
          border-top:1px solid #1a2540; padding-top:14px; display:block; }
#bbox-bridge textarea {
    background:#060d1e !important; border:1px solid #1e3a5f !important;
    border-radius:8px !important; color:#38bdf8 !important;
    font-family:'JetBrains Mono',monospace !important;
    font-size:11px !important; resize:none !important; }
input[type=range] { accent-color:#3b82f6 !important; }
label { color:#94a3b8 !important; font-size:11px !important;
        text-transform:uppercase; letter-spacing:0.07em; }
#run-btn {
    background:linear-gradient(135deg,#2563eb,#1d4ed8) !important;
    border:none !important; border-radius:8px !important;
    color:white !important; font-weight:700 !important;
    font-family:'JetBrains Mono',monospace !important;
    letter-spacing:0.06em; height:46px !important;
    box-shadow:0 4px 20px rgba(37,99,235,0.35) !important; }
#run-btn:hover { background:linear-gradient(135deg,#1d4ed8,#1e40af) !important; }
#run-btn:disabled { opacity:0.35 !important; }
#log-box textarea {
    background:#040a14 !important; border:1px solid #1a2540 !important;
    border-radius:8px !important; color:#4ade80 !important;
    font-family:'JetBrains Mono',monospace !important;
    font-size:11px !important; line-height:1.75 !important; resize:none !important; }
#map-panel { border-radius:12px; overflow:hidden; }
#map-panel iframe { border-radius:12px; border:none; }
"""


# ─────────────────────────────────────────────────────────────
# Pipeline generator
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    bbox_json: str,
    building_threshold: float,
    overlap_threshold: float,
    progress=gr.Progress(track_tqdm=False),
):
    if not bbox_json or bbox_json.strip() == "":
        yield "⚠  No area selected — draw a rectangle on the map first.", _EMPTY_MAP
        return
    try:
        bbox  = json.loads(bbox_json)
        north = bbox["north"]; south = bbox["south"]
        east  = bbox["east"];  west  = bbox["west"]
    except Exception:
        yield "⚠  Could not parse bounding box — please redraw.", _EMPTY_MAP
        return

    log_lines = []
    def log(msg):
        log_lines.append(msg)
        return "\n".join(log_lines)

    # Stage 1 — CRS
    progress(0.02, desc="Converting coordinates…")
    t = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
    tl_x, tl_y = t.transform(west, north)
    br_x, br_y = t.transform(east, south)
    area_m2 = abs(br_x - tl_x) * abs(tl_y - br_y)
    yield log(f"📐  Area: {area_m2/1e6:.4f} km²"), _EMPTY_MAP
    yield log(f"🔁  EPSG:4326 → EPSG:3763 done"), _EMPTY_MAP

    # Stage 2 & 3 — Tiles (cached if bbox unchanged)
    key = _bbox_key(west, south, east, north)
    if key in _tile_cache:
        satellite_image, res = _tile_cache[key]
        h, w = satellite_image.shape[:2]
        progress(0.55, desc="Tiles loaded from cache.")
        yield log(f"⚡  Tile cache hit — skipping download ({w}×{h} px, {res:.3f} m/px)"), _EMPTY_MAP
    else:
        progress(0.06, desc="Connecting to WMTS…")
        yield log("📡  Connecting to DGT WMTS satellite service…"), _EMPTY_MAP
        wmts_url = ("https://cartografia.dgterritorio.gov.pt/ortos2018/service"
                    "?service=WMTS&request=GetCapabilities")
        wmts   = WebMapTileService(wmts_url)
        matrix = wmts.tilematrixsets["PTTM_06"].tilematrix["14"]
        col_min, col_max, row_min, row_max = get_tile_indices(tl_x, br_y, br_x, tl_y, matrix)
        total_tiles = (col_max+1-col_min) * (row_max+1-row_min)
        yield log(f"🛰   Service ready — {total_tiles} tile(s) to download"), _EMPTY_MAP

        def tile_cb(done, total):
            progress(0.10 + 0.45*(done/total), desc=f"Downloading tiles… {done}/{total}")

        satellite_image, res, _ = retrieve_satelite_image(
            (tl_x, tl_y), (br_x, br_y), progress_cb=tile_cb)
        _tile_cache.clear()          # keep memory bounded — only one region at a time
        _tile_cache[key] = (satellite_image, res)
        h, w = satellite_image.shape[:2]
        yield log(f"✅  Satellite image ready  ({w}×{h} px, {res:.3f} m/px)"), _EMPTY_MAP

    # Stage 4 — OSM
    progress(0.57, desc="Fetching OSM buildings…")
    yield log("🗺   Querying Overpass API…"), _EMPTY_MAP
    geojson = get_osm_buildings((tl_x, tl_y), (br_x, br_y))
    yield log(f"✅  OSM done — {len(geojson['features'])} footprint(s)"), _EMPTY_MAP

    # Stage 5 — YOLO
    progress(0.68, desc="Running YOLO inference…")
    yield log(f"🤖  Running detector  (conf≥{building_threshold:.2f}, iou≤{overlap_threshold:.2f})…"), _EMPTY_MAP
    img_bgr   = cv2.cvtColor(satellite_image, cv2.COLOR_RGB2BGR)
    buildings = retrieve_prediction_list(img_bgr, (tl_x, tl_y), res,
                                         building_threshold, overlap_threshold, model)
    total_planes = sum(len(b.planes_in_physical_dimensions) for b in buildings)
    yield log(f"✅  {len(buildings)} building(s), {total_planes} roof plane(s)"), _EMPTY_MAP

    # Stage 6 — Render
    progress(0.88, desc="Rendering map…")
    yield log("🗾  Compositing layers…"), _EMPTY_MAP
    map_html = build_map_html(geojson, buildings, satellite_image,
                               (tl_x, tl_y), (br_x, br_y))
    progress(1.0, desc="Done!")
    yield log("🎉  All done! Map is live →"), map_html


# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Building Topology Detector", css=CSS, theme=gr.themes.Base()) as demo:

    with gr.Column(elem_id="hdr"):
        gr.HTML("<h1>🛰 Building Topology Detector</h1>")
        gr.HTML("<p>YOLO · WMTS · OSM · EPSG:3763 — draw a region on the map to begin</p>")

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=340, elem_id="left-panel"):
            gr.HTML('<span class="slabel" style="border-top:none;padding-top:0">① Select Region</span>')
            gr.HTML(SELECTION_MAP_HTML)

            bbox_bridge = gr.Textbox(
                label="Selected bounding box (WGS84 JSON)",
                placeholder="Draw a rectangle above to populate…",
                interactive=True, lines=2, elem_id="bbox-bridge",
            )

            gr.HTML('<span class="slabel">② Detection Parameters</span>')
            building_slider = gr.Slider(0.10, 0.95, step=0.05, value=0.50,
                                        label="Building Confidence Threshold")
            overlap_slider  = gr.Slider(0.10, 0.95, step=0.05, value=0.50,
                                        label="Overlap (IoU) Threshold")

            run_btn = gr.Button("▶  Run Detection", variant="primary", elem_id="run-btn")

            gr.HTML('<span class="slabel">③ Progress Log</span>')
            log_box = gr.Textbox(label="", interactive=False, lines=10, max_lines=10,
                                 placeholder="Logs will stream here…", elem_id="log-box")

        with gr.Column(scale=3, elem_id="map-panel"):
            map_display = gr.HTML(value=_EMPTY_MAP)

    run_btn.click(
        fn=run_pipeline,
        inputs=[bbox_bridge, building_slider, overlap_slider],
        outputs=[log_box, map_display],
    )

    # Register the postMessage listener only after the full DOM exists
    demo.load(fn=None, js=_BBOX_LISTENER_JS)

demo.launch()
