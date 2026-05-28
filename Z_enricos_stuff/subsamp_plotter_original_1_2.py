# region INPUT DATA

case1_3surf_from2 = {'epi+lv': {'chamfer': {'epi': (0.7092031567088608,
                                                      0.11673735865301972),
                                              'lv': (0.5854551849367088,
                                                     0.17480813221745317),
                                              'rv': (0.9561444312658229,
                                                     0.2920981936949682)},
                                  'haussdorff': {'epi': (3.914181552428656,
                                                         1.2402439028291188),
                                                 'lv': (2.990805703972272,
                                                        1.413963315424487),
                                                 'rv': (5.41473785631682,
                                                        4.25985351138985)}},
                       'epi+rv': {'chamfer': {'epi': (0.691080825949367,
                                                      0.11849936733691917),
                                              'lv': (1.5351497496835442,
                                                     0.7180643606140134),
                                              'rv': (0.595945798164557,
                                                     0.13561569404711252)},
                                  'haussdorff': {'epi': (3.806364162296959,
                                                         1.3784348545188543),
                                                 'lv': (5.766537578555323,
                                                        3.287305614798532),
                                                 'rv': (3.065225364304638,
                                                        1.5514186094146118)}},
                       'lv+rv': {'chamfer': {'epi': (1.0345228656329113,
                                                     0.2638989299909345),
                                             'lv': (0.5924619794303797,
                                                    0.18500310109844545),
                                             'rv': (0.6332129265822785,
                                                    0.16479339405346935)},
                                 'haussdorff': {'epi': (5.045293621841072,
                                                        1.6630066111258117),
                                                'lv': (3.0681276982919985,
                                                       1.314267353997424),
                                                'rv': (3.237749653135219,
                                                       1.510701755919786)}}}

case2_sdf_filter = {
                     'sdf_thresh_0.5': {'chamfer': {'epi': (0.7145942419620254,
                                                           0.1257650258694242),
                                                   'lv': (0.6212357156962026,
                                                          0.18462225433283994),
                                                   'rv': (0.6333712522151899,
                                                          0.15465280203431656)},
                                       'haussdorff': {'epi': (3.8483180428694297,
                                                              1.2051642950075834),
                                                      'lv': (3.171681821038409,
                                                             1.4528430913715857),
                                                      'rv': (3.28528885350068,
                                                             1.7128410785528354)}},
                     'sdf_thresh_0.1': {'chamfer': {'epi': (0.7169763041772153,
                                                           0.12683956197652255),
                                                   'lv': (0.6226124833544303,
                                                          0.185451846977958),
                                                   'rv': (0.631819262721519,
                                                          0.15473136885845812)},
                                       'haussdorff': {'epi': (3.8514156337723335,
                                                              1.2205548690883046),
                                                      'lv': (3.168014149406292,
                                                             1.4362995949814554),
                                                      'rv': (3.272456964913765,
                                                             1.663670874713315)}},
                     'sdf_thresh_0.002': {'chamfer': {'epi': (0.7262549403797468,
                                                             0.12963849008330483),
                                                     'lv': (0.6294796100632911,
                                                            0.18648380224502573),
                                                     'rv': (0.6359728195569619,
                                                            0.1554261077340442)},
                                         'haussdorff': {'epi': (3.8824782875914203,
                                                                1.2176169426969674),
                                                        'lv': (3.1892946147210175,
                                                               1.442919911668677),
                                                        'rv': (3.260202999635223,
                                                               1.6354548643832065)}},
                     'sdf_thresh_0.001': {'chamfer': {'epi': (0.7352496483544304,
                                                             0.13218018206174698),
                                                     'lv': (0.6358058065822785,
                                                            0.18745404710993382),
                                                     'rv': (0.6454721379746836,
                                                            0.15605634510927005)},
                                         'haussdorff': {'epi': (3.9824873863835317,
                                                                1.2544574062290665),
                                                        'lv': (3.2058769426242417,
                                                               1.4057904842850621),
                                                        'rv': (3.3176402675432635,
                                                               1.592683023667524)}},
                     'sdf_thresh_0.0005': {'chamfer': {'epi': (0.755590103164557,
                                                              0.13345338173954746),
                                                      'lv': (0.6495865359493671,
                                                             0.18749026926157544),
                                                      'rv': (0.6683763094303797,
                                                             0.15985119845783474)},
                                          'haussdorff': {'epi': (4.198465586093311,
                                                                 1.3329287268593846),
                                                         'lv': (3.331420881141629,
                                                                1.436412875909678),
                                                         'rv': (3.4948540020518544,
                                                                1.62464918512733)}},                 
                      }

original = {'chamfer': {'epi': (0.71304,
                                0.12139),
                        'lv': (0.62123,
                                0.18620),
                        'rv': (0.63375,
                                0.13323)},
            'haussdorff': {'epi': (3.86309,
                                    1.20920),
                            'lv': (3.17575,
                                1.45383),
                            'rv': (3.29959,
                                1.40665)}}


# endregion

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
    comparison_dict=case1_3surf_from2,
    title='Chamfer - Original vs Surface Inference',
)

make_comparison_plot(
    metric_name='haussdorff',
    original=original,
    comparison_dict=case1_3surf_from2,
    title='Haussdorff - Original vs Surface Inference',
)


# --------------------------------------
# CASE 2: sdf threshold filtering
# --------------------------------------
make_comparison_plot(
    metric_name='chamfer',
    original=original,
    comparison_dict=case2_sdf_filter,
    title='Chamfer - Original vs SDF Threshold'
)

make_comparison_plot(
    metric_name='haussdorff',
    original=original,
    comparison_dict=case2_sdf_filter,
    title='Haussdorff - Original vs SDF Threshold'
)