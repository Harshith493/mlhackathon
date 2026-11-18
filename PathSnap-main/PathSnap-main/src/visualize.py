"""
visualize.py - Interactive Folium geographical map generation
for PathSnap Map-Matching and Route Reconstruction.
"""

import folium
from folium import plugins


def create_folium_map(
    track,
    arcs,
    nodes,
    true_route_arcs=None,
    classifier_pred_arcs=None,
    hmm_pred_arcs=None,
    trip_id="00000000",
    out_html=None,
):
    """
    Builds a multi-layer interactive Folium map comparing raw GPS, road network,
    ground-truth route, classifier-only predictions, and HMM reconstructed route.
    """
    mean_lat = float(np_mean := track[:, 1].mean())
    mean_lon = float(track[:, 0].mean())

    fmap = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 1. Raw GPS Observations Layer
    gps_layer = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    for i in range(len(track)):
        lon, lat = track[i, 0], track[i, 1]
        t = track[i, 2] if track.shape[1] > 2 else i
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#1E88E5",
            fill=True,
            fill_color="#1E88E5",
            fill_opacity=0.8,
            popup=f"GPS #{i}<br>Time: {t:.1f}s<br>Coords: ({lat:.5f}, {lon:.5f})",
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    # 2. Road Network Geometry Layer (Local bounding area)
    min_lon, max_lon = track[:, 0].min() - 0.005, track[:, 0].max() + 0.005
    min_lat, max_lat = track[:, 1].min() - 0.005, track[:, 1].max() + 0.005

    network_layer = folium.FeatureGroup(name="Road Network (Arcs)", show=False)
    for arc_id in range(len(arcs)):
        u, v = arcs[arc_id]
        u_lon, u_lat = nodes[u, 0], nodes[u, 1]
        v_lon, v_lat = nodes[v, 0], nodes[v, 1]
        if min_lon <= u_lon <= max_lon and min_lat <= u_lat <= max_lat:
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color="#78909C",
                weight=1.5,
                opacity=0.6,
                popup=f"Arc #{arc_id}<br>Nodes: {u} -> {v}",
            ).add_to(network_layer)
    network_layer.add_to(fmap)

    # 3. Ground Truth Reference Route Layer
    if true_route_arcs is not None:
        gt_layer = folium.FeatureGroup(name="Ground Truth Route (Reference)", show=True)
        # Unique ordered segments
        seen_arcs = []
        for a in true_route_arcs:
            if not seen_arcs or seen_arcs[-1] != a:
                seen_arcs.append(a)

        for a in seen_arcs:
            if 0 <= a < len(arcs):
                u, v = arcs[a]
                u_lon, u_lat = nodes[u, 0], nodes[u, 1]
                v_lon, v_lat = nodes[v, 0], nodes[v, 1]
                folium.PolyLine(
                    locations=[[u_lat, u_lon], [v_lat, v_lon]],
                    color="#2E7D32",
                    weight=4.5,
                    opacity=0.9,
                    popup=f"Ground Truth Arc #{a}",
                ).add_to(gt_layer)
        gt_layer.add_to(fmap)

    # 4. Classifier-Only Prediction Layer
    if classifier_pred_arcs is not None:
        clf_layer = folium.FeatureGroup(name="Classifier-Only Predictions", show=False)
        seen_clf = []
        for a in classifier_pred_arcs:
            if not seen_clf or seen_clf[-1] != a:
                seen_clf.append(a)

        for a in seen_clf:
            if 0 <= a < len(arcs):
                u, v = arcs[a]
                u_lon, u_lat = nodes[u, 0], nodes[u, 1]
                v_lon, v_lat = nodes[v, 0], nodes[v, 1]
                folium.PolyLine(
                    locations=[[u_lat, u_lon], [v_lat, v_lon]],
                    color="#D32F2F",
                    weight=3.5,
                    dash_array="6, 6",
                    opacity=0.85,
                    popup=f"Classifier Pred Arc #{a}",
                ).add_to(clf_layer)
        clf_layer.add_to(fmap)

    # 5. HMM Reconstructed Route Layer
    if hmm_pred_arcs is not None:
        hmm_layer = folium.FeatureGroup(name="HMM Reconstructed Route (Viterbi)", show=True)
        seen_hmm = []
        for a in hmm_pred_arcs:
            if not seen_hmm or seen_hmm[-1] != a:
                seen_hmm.append(a)

        for a in seen_hmm:
            if 0 <= a < len(arcs):
                u, v = arcs[a]
                u_lon, u_lat = nodes[u, 0], nodes[u, 1]
                v_lon, v_lat = nodes[v, 0], nodes[v, 1]
                folium.PolyLine(
                    locations=[[u_lat, u_lon], [v_lat, v_lon]],
                    color="#00BCD4",
                    weight=4.0,
                    opacity=0.95,
                    popup=f"HMM Reconstructed Arc #{a}",
                ).add_to(hmm_layer)
        hmm_layer.add_to(fmap)

    # Floating Legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px; left: 30px; width: 280px; height: auto;
        background-color: white; z-index:9999; font-size:13px;
        border: 2px solid #ccc; border-radius: 8px; padding: 12px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.25);
        font-family: Arial, sans-serif;
    ">
        <h4 style="margin: 0 0 8px 0; font-size: 14px; color: #1a237e;">
            <b>PathSnap Map-Matching ({trip_id})</b>
        </h4>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#1E88E5; margin-right:8px;"></span>
            Raw GPS Points ({len(track)})
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:18px; height:4px; background:#2E7D32; vertical-align:middle; margin-right:8px;"></span>
            Ground Truth Route
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:18px; height:4px; border-top: 2px dashed #D32F2F; vertical-align:middle; margin-right:8px;"></span>
            Classifier-Only Prediction
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:18px; height:4px; background:#00BCD4; vertical-align:middle; margin-right:8px;"></span>
            HMM Reconstructed Route
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:18px; height:2px; background:#78909C; vertical-align:middle; margin-right:8px;"></span>
            Road Network Arcs
        </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

    if out_html:
        fmap.save(out_html)
    return fmap

