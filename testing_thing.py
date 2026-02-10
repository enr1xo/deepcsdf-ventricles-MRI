import numpy as np
import plotly.graph_objects as go

for col in range(3, 6):
    SDF_COL = col            # 3=epi, 4=lv, 5=rv

    patients_list = ["AF001", "AF002_P2", "AF003_P2",
                    "AF004_P1", "AF005_P1R", "AF006",
                    "AF007_P1", "AF008_P1"]

    for i in range(len(patients_list)):
        patient_nbr = i # from 0 to 7
        patient = patients_list[patient_nbr]

        # ====== INPUT/OUTPUT ======
        NPY_PATH = f"/mnt/c/Users/e.rizzardi/OneDrive/Desktop/AF_patients/single_patients_100000pts_npy/{patient}-epi_lv_rv_100000_coords_and_sdf.npy"

        if SDF_COL == 3:
            OUT_HTML = f"/mnt/c/Users/e.rizzardi/OneDrive/Desktop/samplings_and_sdf/{patient}/{patient}_sdf_EPI_plotly.html"
        elif SDF_COL == 4:
            OUT_HTML = f"/mnt/c/Users/e.rizzardi/OneDrive/Desktop/samplings_and_sdf/{patient}/{patient}_sdf_LV_plotly.html" 
        elif SDF_COL == 5:
            OUT_HTML = f"/mnt/c/Users/e.rizzardi/OneDrive/Desktop/samplings_and_sdf/{patient}/{patient}_sdf_RV_plotly.html"


        # OUT_HTML = "/mnt/c/Users/e.rizzardi/OneDrive/Desktop/AF008_P1_sdf_epi_plotly.html"

        # ====== PARAMS ======
        MAX_POINTS = 100000     # riduci se il browser è lento
        NEAR_EPS = 0.2        # evidenzia superficie (|sdf| < eps)

        data = np.load(NPY_PATH)
        coords = data[:, :3].astype(np.float32)
        sdf = data[:, SDF_COL].astype(np.float32)

        # downsample per performance (mantieni più near-surface)
        near = np.where(np.abs(sdf) < NEAR_EPS)[0]
        far = np.where(np.abs(sdf) >= NEAR_EPS)[0]

        rng = np.random.default_rng(0)
        n_near = min(len(near), int(MAX_POINTS * 0.85))
        n_far  = min(len(far),  MAX_POINTS - n_near)

        near_idx = rng.choice(near, size=n_near, replace=False) if len(near) else np.array([], dtype=int)
        far_idx  = rng.choice(far,  size=n_far,  replace=False) if len(far) else np.array([], dtype=int)
        idx = np.concatenate([near_idx, far_idx])

        x, y, z = coords[idx].T
        s = sdf[idx]

        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=1.5,
                color=s,
                colorscale="Jet",
                colorbar=dict(title="SDF"),
                opacity=0.9
            )
        ))

        if SDF_COL == 3:
            region = "EPI"
        elif SDF_COL == 4:
            region = "LV" 
        elif SDF_COL == 5:
            region = "RV"

        fig.update_layout(
            title=f"SDF region={region} (points shown={len(idx)})",
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z",
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        fig.write_html(OUT_HTML, include_plotlyjs=True)
        print("HTML salvato in:", OUT_HTML)
