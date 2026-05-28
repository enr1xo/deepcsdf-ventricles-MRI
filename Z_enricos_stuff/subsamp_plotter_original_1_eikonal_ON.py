#region INPUT DATA
original = {'chamfer': {'epi': (0.701,
                                0.12139),
                        'lv': (0.61,
                                0.18),
                        'rv': (0.63,
                                0.13)},
            'haussdorff': {'epi': (3.745,
                                    1.261),
                            'lv': (2.98,
                                1.43),
                            'rv': (3.11,
                                1.148)}}

case1_3surf_from2_eik_on = {'epi+lv': {'chamfer': {'epi': (0.7053542081012658,
                                               0.11778663284876018),
                                       'lv': (0.5721588099999999,
                                              0.17058959657899575),
                                       'rv': (0.9612435581012657,
                                              0.2907973201344895)},
                           'haussdorff': {'epi': (3.830013805359631,
                                                  1.2720676889394125),
                                          'lv': (2.8055761867309355,
                                                 1.3759642317601901),
                                          'rv': (5.296671824612109,
                                                 4.19154059089757)}},
                'epi+rv': {'chamfer': {'epi': (0.68444035778481,
                                               0.11738647092936311),
                                       'lv': (1.6302054038607592,
                                              0.7447942681569436),
                                       'rv': (0.5869920355063292,
                                              0.13342731941832603)},
                           'haussdorff': {'epi': (3.6736922683113398,
                                                  1.3023613399803784),
                                          'lv': (5.894618932743166,
                                                 3.310040985582451),
                                          'rv': (2.9156015086100857,
                                                 1.360644664440493)}},
                'lv+rv': {'chamfer': {'epi': (1.0484775059493672,
                                              0.2784731082593596),
                                      'lv': (0.5778236359493671,
                                             0.17569861841462248),
                                      'rv': (0.6246041885443039,
                                             0.15541205661268176)},
                          'haussdorff': {'epi': (5.060485245776468,
                                                 1.801300856454586),
                                         'lv': (2.898431553143171,
                                                1.3039345190020277),
                                         'rv': (3.172698871883992,
                                                1.3753128245001343)}}}
#endregion

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PLOT FUNCTION
# ============================================================
def make_comparison_plot(metric_name, original, comparison_dict, title):

    surfaces = ['epi', 'lv', 'rv']
    x = np.arange(len(surfaces))

    labels = ['original'] + list(comparison_dict.keys())

    width = 0.08

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, label in enumerate(labels):

        if label == 'original':
            data = original[metric_name]
        else:
            data = comparison_dict[label][metric_name]

        means = [data[s][0] for s in surfaces]
        stds = [data[s][1] for s in surfaces]

        offset = (i - (len(labels) - 1) / 2) * width

        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt='o',
            capsize=5,
            label=label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(['epi', 'lv_endo', 'rv_endo'])

    ax.set_ylabel(f"{metric_name} (mm)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# PLOTS
# ============================================================

# --------------------------------------
# CASE 1: infer one surface from two
# --------------------------------------
make_comparison_plot(
    metric_name='chamfer',
    original=original,
    comparison_dict=case1_3surf_from2_eik_on,
    title='Chamfer - Original vs Surface Inference - eikonal ON',
)

make_comparison_plot(
    metric_name='haussdorff',
    original=original,
    comparison_dict=case1_3surf_from2_eik_on,
    title='Haussdorff - Original vs Surface Inference - eikonal ON',
)


# --------------------------------------
# CASE 2: sdf threshold filtering
# --------------------------------------
# make_comparison_plot(
#     metric_name='chamfer',
#     original=original,
#     comparison_dict=case2_sdf_filter,
#     title='Chamfer - Original vs SDF Threshold'
# )

# make_comparison_plot(
#     metric_name='haussdorff',
#     original=original,
#     comparison_dict=case2_sdf_filter,
#     title='Haussdorff - Original vs SDF Threshold'
# )