import pyvista as pv
from pathlib import Path
from config import PATIENT_MESHES_DIR, RECONSTRUCTED_MESHES_DIR, ATRIA_TAGS_METADATA
import numpy as np
from utils.align_atrial_mesh import apply_icp_result


# =============================================================== #
# VOLUMETRIC MESH 
# =============================================================== #
# patient = "LEU_NORM_0832"

# patient_meshes_dir = Path("/home/davidenava_linux/DATASETS/AtrialGeometriesDataProcessed")
# mesh = pv.read(patient_meshes_dir / f"{patient}/{patient}.vtu")

# # try interactively
# plotter = pv.Plotter(window_size=[2000, 2000])
# plotter.add_mesh(mesh, color = "white", opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.camera_position = [(-121136.86699005571, -102974.16596629776, -275893.00992146623),
#  (-34985.71306974535, -166908.90131753965, -401624.6852106305),
#  (-0.24535032287060562, 0.7858209420643972, -0.5677047349461803)]
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.close()

# # save screenshot with transparent background
# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(mesh, color = "white", opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.camera_position =[(-121136.86699005571, -102974.16596629776, -275893.00992146623),
#  (-34985.71306974535, -166908.90131753965, -401624.6852106305),
#  (-0.24535032287060562, 0.7858209420643972, -0.5677047349461803)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-volumetric-mesh-solid.png", transparent_background=True)
# plotter.close()

# R = mesh.field_data["aligntoAF069-rotation"]
# t = mesh.field_data["aligntoAF069-traslation"]
# c = mesh.field_data["scale-centroid"]
# r = mesh.field_data["scale-radius"]

# mesh.points = apply_icp_result(R,t, (mesh.points - c) / r)

# mesh.points *= r

# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(mesh, color = "white", opacity = 1.0, show_edges=True, line_width=0.01)
# plotter.camera_position = [(-203942.26164071326, 14489.584117118184, 179454.37546262395),
#                            (27440.068707785256, -13344.81572577331, 2187.7828968081194),
#                            (-0.3802976541693145, 0.6985926216629214, -0.6060874880670636)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-volumetric-mesh.png", transparent_background=True)
# plotter.close()

# mesh_reconstructed = pv.read("reconstructed_AF009_P2R_version_82_epicardium.vtp")
# plotter = pv.Plotter(window_size=[1280,1280], off_screen=True)
# plotter.add_mesh(mesh_reconstructed, color = "white", opacity = 1.0, show_edges=True, line_width=0.01)
# plotter.background_color = None
# plotter.camera_position = [(-422.91183409819524, 47.02461156771938, 246.6199194031598),
#  (24.17760632591991, -11.779115610423997, -1.7062562022323533),
#  (-0.32860612577400417, 0.5959126749992583, -0.7327387650991504)]
# plotter.show(auto_close=False)
# print(plotter.camera_position)
# plotter.screenshot(f"{patient}-epi-reconstructed.png", transparent_background=True)
# plotter.close()



# # ============ surfaces extracted with tags ================ #
# if hasattr(mesh, "cell_data"):
#     if "elemTag" in mesh.cell_data:
#         elemTags = mesh.cell_data["elemTag"]
#         elemtagskey =  "elemTag"
#     elif "elemTags" in mesh.cell_data:
#         elemTags = mesh.cell_data["elemTags"]
#         elemtagskey =  "elemTags"

# mesh_tags = set(elemTags) # for lookup 

# from atria_data_preparing import split_cell_data_tags
# import numpy as np

# split_tags = split_cell_data_tags(mesh_tags) # TODO: add extracting tags info from reading the .aug file directly?

# RA_endo_tags = split_tags["RA_ENDO_TAGS"]

# LA_endo_tags = split_tags["LA_ENDO_TAGS"]

# RA_epi_tags = split_tags["RA_EPI_TAGS"]

# LA_epi_tags = split_tags["LA_EPI_TAGS"]

# # # =============================================================== #
# # #  Extract surfaces
# # # =============================================================== #
# whole_surface = mesh.extract_surface()
# surface_elemTags = whole_surface.cell_data[elemtagskey] 

# # ===== epicardium ===== #
# # epicardium: 97 is tag for "RA_FO", which is part of endocardium also but it fills a hole in epi connecting it to the left epicardium sometimes
# # if I don't put it may happen I extract two disconnected components for some particular patients [...]
# surf = whole_surface.extract_cells( np.isin( surface_elemTags, RA_epi_tags + LA_epi_tags + [97]) )
# surf = surf.connectivity(extraction_mode = 'largest')  
# epicardium_surface = surf.extract_surface()

# # ===== right atrium endocardium ===== #
# surf = whole_surface.extract_cells( np.isin( surface_elemTags, RA_endo_tags) )
# surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
# RA_endo_surface = surf.extract_surface()

# # ===== left atrium endocardium ===== #
# surf = whole_surface.extract_cells( np.isin( surface_elemTags, LA_endo_tags) )
# surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
# LA_endo_surface = surf.extract_surface()



# # try interactively
# plotter = pv.Plotter(window_size=[2000, 2000])
# plotter.add_mesh(RA_endo_surface, color = (68,114,196), opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.camera_position = [(-129991.6050501295, -90531.85579298022, -39815.972965522946),
#  (54681.0444620354, -227581.63906221386, -309332.9851779291),
#  (-0.24535032287060562, 0.7858209420643972, -0.5677047349461803)]
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.close()

# # save screenshot with transparent background
# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(epicardium_surface, color = "lightgray", opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.camera_position = [(-129991.6050501295, -90531.85579298022, -39815.972965522946),
#  (54681.0444620354, -227581.63906221386, -309332.9851779291),
#  (-0.24535032287060562, 0.7858209420643972, -0.5677047349461803)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-extracted-epi.png", transparent_background=True)
# plotter.close()

# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(RA_endo_surface, color = (68,114,196), opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.camera_position = [(-129991.6050501295, -90531.85579298022, -39815.972965522946),
#  (54681.0444620354, -227581.63906221386, -309332.9851779291),
#  (-0.24535032287060562, 0.7858209420643972, -0.5677047349461803)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-extracted-ra.png", transparent_background=True)
# plotter.close()

# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(LA_endo_surface, color = (255,151,103), opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.camera_position = [(-129991.6050501295, -90531.85579298022, -39815.972965522946),
#  (54681.0444620354, -227581.63906221386, -309332.9851779291),
#  (-0.24535032287060562, 0.7858209420643972, -0.5677047349461803)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-extracted-la.png", transparent_background=True)
# plotter.close()




# plotter = pv.Plotter(window_size=[1720,1280])
# plotter.add_mesh(epicardium_surface, color = "lightgray", opacity = 0.8)
# plotter.add_mesh(RA_endo_surface, color = "skyblue", opacity = 0.8)
# plotter.add_mesh(LA_endo_surface, color = "red", opacity = 0.8)
# # plotter.camera_position = [(-69694.6085575019, -166835.31454372327, 198549.69480718492),
# #  (25694.152592809456, -146074.99422346364, 24718.368900526904),
# #  (-0.4557435189567757, 0.8781873664364168, -0.14520604106016416)]
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.screenshot(f"{patient}-epicardium-la-ra-raw.png")




# ============ processed surfaces ================ #

# epicardium = pv.read(PATIENT_MESHES_DIR / f"{patient}/epicardium-processed.vtp")
# LA_endo = pv.read(PATIENT_MESHES_DIR / f"{patient}/la_endo-processed.vtp")
# RA_endo = pv.read(PATIENT_MESHES_DIR / f"{patient}/ra_endo-processed.vtp")


# # try interactively
# plotter = pv.Plotter(window_size=[2000, 2000])
# plotter.add_mesh(LA_endo, color = "white", opacity = 1.0, show_edges=True) #, line_width=0.01)
# # plotter.camera_position = [(0.4467329410674157, -0.17509100216971896, 2.849232330208489),
# #  (0.20103319793731003, 0.3019052849592955, -0.4329513608003727),
# #  (-0.5840803542013553, 0.7958926238181849, 0.159389683442262)]
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.close()

# # save screenshot with transparent background
# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(epicardium, color = "white", opacity = 1.0, show_edges=True) #, line_width=0.01)
# plotter.camera_position =[(-2.1396565180513893, -0.6822416585530134, 2.740462211191305),
#  (0.5800713802128634, 0.04271095740090308, -0.5286301208575954),
#  (-0.3862621917973392, 0.9147457321350548, -0.1184979524237365)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-processed-epi-white-edges.png", transparent_background=True)
# plotter.close()

# # save screenshot with transparent background
# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(RA_endo, color = "white", opacity = 1.0, show_edges=True) #, line_width=0.01)
# plotter.camera_position = [(-3.4274233201424784, 1.1329721626100688, 0.976372183845008),
#  (0.17152472760237186, -0.41642244148663315, 0.05941617185530429),
#  (-0.16250924398853966, 0.19495859684865785, -0.967254822233043)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-processed-ra-white-edges.png", transparent_background=True)
# plotter.close()

# # save screenshot with transparent background
# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(LA_endo, color = "white", opacity = 1.0, show_edges=True) #, line_width=0.01)
# plotter.camera_position =[(2.392032830851917, -1.7124428855506673, 0.9434242032489181),
#  (0.20242251747256876, 0.5484840435513089, -0.13102435300583348),
#  (-0.4637190103796397, -0.02826542623360717, 0.8855313349014592)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-processed-la-white-edges.png", transparent_background=True)
# plotter.close()



# epicardium.points *= r
# LA_endo.points *= r
# RA_endo.points *= r

# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(epicardium, color="lightgray", opacity=0.5)
# plotter.add_mesh(LA_endo, color="red", opacity=0.8)
# plotter.add_mesh(RA_endo, color="skyblue", opacity=0.8)
# plotter.camera_position = [(-252532.55101389796, 20334.808084125398, 216680.3599014453),
#  (27440.068707785256, -13344.81572577331, 2187.7828968081194),
#  (-0.3802976541693145, 0.6985926216629214, -0.6060874880670636)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-epicardium-la-ra-processed.png", transparent_background=True)
# plotter.close()


# plotter = pv.Plotter(window_size=[2000,2000])
# plotter.add_mesh(epicardium, color = "white", opacity = 1.0, show_edges=True, line_width=0.01)
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.screenshot(f"{patient}-epicardium-processed.png")

# plotter = pv.Plotter(window_size=[1720,1280])
# plotter.add_mesh(LA_endo, color = "white", opacity = 1.0, show_edges=True, line_width=0.01)
# plotter.camera_position = [(1.8000032144242024, 0.6011739383004858, 2.2273821525722957),
#  (0.2809552709247972, 0.14175695830885446, -0.0167176958260283),
#  (-0.5089437115022734, 0.8434754657885282, 0.1718296747803349)]
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.screenshot(f"{patient}-la-endp-processed.png")

# plotter = pv.Plotter(window_size=[1720,1280])
# plotter.add_mesh(RA_endo, color = "white", opacity = 1.0, show_edges=True, line_width=0.01)
# plotter.camera_position = [(0.3289782058669293, 1.7853429965662098, -1.2131484085191468),
#  (-0.29749090047336596, -0.42889078115109663, -0.14352771938395598),
#  (-0.6548260523691133, 0.47085101545412444, 0.5911870790067247)]
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.screenshot(f"{patient}-ra-endo-processed.png")









# ============== #
# ALIGN ANOTHER MESH TO THE FIRST
# ================ #

# import numpy as np

# patient2 = "LEU_NORM_0778"

# mesh2 =  pv.read(PATIENT_MESHES_DIR / f"{patient2}/{patient2}.vtu")

# mesh.points -= np.mean(mesh.points, axis=0)
# mesh2.points -= np.mean(mesh2.points, axis=0)

# # try interactively
# plotter = pv.Plotter(window_size=[2000, 2000])
# plotter.add_mesh(mesh, color = (0,121,52), opacity = 0.3)
# plotter.add_mesh(mesh2, color = "white", opacity = 0.5)
# # plotter.camera_position = [(-67140.0114075277, 170961.65869949464, 242608.3771551897),
# #  (4974.983193793472, -26041.625641130282, 4665.399871338299),
# #  (0.12982117838276983, 0.7827113671761254, -0.6086948146128646)]
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.close()

# # save screenshot with transparent background
# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(mesh, color = (0,121,52), opacity = 0.5)
# plotter.add_mesh(mesh2, color = "white", opacity = 0.8)
# plotter.camera_position =[(-176084.50715709926, -73586.45447494964, 201973.35527938238),
#  (32011.329789855685, -11551.807922075775, -27683.58375539257),
#  (-0.26283669641760615, 0.9645804922437923, 0.02239073466866874)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-{patient2}-notaligned.png", transparent_background=True)
# plotter.close()


# # align with ICP
# from utils.align_atrial_mesh import align_to_reference_mesh

# mesh, R, t = align_to_reference_mesh(
#     mesh.extract_surface(),
#     mesh2.extract_surface(),
#     max_iter=100,
#     verbose_out=True
# )

# # try interactively
# plotter = pv.Plotter(window_size=[2000, 2000])
# plotter.add_mesh(mesh, color = (0,121,52), opacity = 0.3)
# plotter.add_mesh(mesh2, color = "white", opacity = 0.5)
# # plotter.camera_position = [(-67140.0114075277, 170961.65869949464, 242608.3771551897),
# #  (4974.983193793472, -26041.625641130282, 4665.399871338299),
# #  (0.12982117838276983, 0.7827113671761254, -0.6086948146128646)]
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.close()

# # save screenshot with transparent background
# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(mesh, color = (0,121,52), opacity = 0.5)
# plotter.add_mesh(mesh2, color = "white", opacity = 0.8)
# plotter.camera_position =[(-176084.50715709926, -73586.45447494964, 201973.35527938238),
#  (32011.329789855685, -11551.807922075775, -27683.58375539257),
#  (-0.26283669641760615, 0.9645804922437923, 0.02239073466866874)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient}-{patient2}-aligned.png", transparent_background=True)
# plotter.close()



# =========================================================== #
# MESH FROM FILE
# =========================================================== #

for t in [0,1/6,2/6,3/6,4/6,5/6,1]:
        
    mesh_name = f"version_114_AF009_P2R-to-LEU_NORM_F004-time={t}"
    mesh = pv.read(f"results/reconstructed/interpolated/{mesh_name}.vtp")

    # # try interactively
    # plotter = pv.Plotter(window_size=[2000, 2000])
    # plotter.add_mesh(mesh, color = "white", opacity = 1.0, show_edges=False) #, line_width=0.01)
    # # plotter.camera_position = [(-129991.6050501295, -90531.85579298022, -39815.972965522946),
    # #  (54681.0444620354, -227581.63906221386, -309332.9851779291),
    # #  (-0.24535032287060562, 0.7858209420643972, -0.5677047349461803)]
    # plotter.show(interactive=True)
    # print(plotter.camera_position)
    # plotter.close()
    # break

    # save screenshot with transparent background
    plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
    plotter.add_mesh(mesh, color = "white", opacity = 1.0, show_edges=False) #, line_width=0.01)
    plotter.camera_position =[(-188.82330388717367, -8.728124190207877, 369.77310709765993),
 (25.40087466858511, -16.27755260044343, -15.06575144006375),
 (-0.4412318652836278, 0.858157445740492, -0.2624504512826848)]
    plotter.background_color = None
    plotter.show(auto_close=False)
    plotter.screenshot(f"{mesh_name}.png", transparent_background=True)
    plotter.close()


# # epi + la + ra from meshes file
# epi = pv.read("results/reconstructed/reconstructed_LEU_NORM_F004_version_114_epicardium-res=128.vtp")
# la = pv.read("results/reconstructed/reconstructed_LEU_NORM_F004_version_114_la_endo-res=128.vtp")
# ra = pv.read("results/reconstructed/reconstructed_LEU_NORM_F004_version_114_ra_endo-res=128.vtp")

# # try interactively
# plotter = pv.Plotter(window_size=[2000, 2000])
# plotter.add_mesh(epi, color = "white", opacity = 0.5, show_edges=False)
# plotter.add_mesh(la, color = "red", opacity = 0.5, show_edges=False) 
# plotter.add_mesh(ra, color = (68,114,196), opacity = 0.5, show_edges=False) 
# # plotter.camera_position = [(-129991.6050501295, -90531.85579298022, -39815.972965522946),
# #  (54681.0444620354, -227581.63906221386, -309332.9851779291),
# #  (-0.24535032287060562, 0.7858209420643972, -0.5677047349461803)]
# plotter.show(interactive=True)
# print(plotter.camera_position)
# plotter.close()

# # save screenshot with transparent background
# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(mesh, color = "white", opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.camera_position =[(262.78852494316413, 243.88501611762456, 223.68414063482945),
#  (13.934397404449376, -4.969111421090091, -25.16998690388525),
#  (0.0, 0.0, 1.0)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{mesh_name}.png", transparent_background=True)
# plotter.close()













# # ============================================================ #
# # RECONSTRUCTED MESHES FROM FILE
# # ============================================================ #
# # retrieve reconstructed mesh
# version = "version_114"
# patient_name = "AF003_P2"
# organ = "epicardium"
# mesh_file = RECONSTRUCTED_MESHES_DIR / f"reconstructed_{patient_name}_{version}_{organ}-res=128.vtp" 
# mesh_pred = pv.read(mesh_file)

# # retrieve original mesh
# patient_dir = PATIENT_MESHES_DIR / patient_name
# mesh_file = next( patient_dir.rglob(f"{organ}-processed.vtp"), None)
# mesh_orig = pv.read(mesh_file)

# # scale both to same range
# scale = mesh_orig.field_data["scale_to_original_range"]
# mesh_orig.points *= scale

# # interactively
# plotter = pv.Plotter(shape=(1, 3), window_size=[1920, 720])
# plotter.subplot(0, 0)
# plotter.add_text(f"Original mesh: patient {patient_name}", font_size=12)
# plotter.add_mesh(mesh_orig, color="lightgray", opacity=1.0)
# plotter.subplot(0, 1)
# plotter.add_text(f"Reconstructed mesh: patient {patient_name}", font_size=12)
# plotter.add_mesh(mesh_pred, color="lightgray", opacity=1.0)
# plotter.subplot(0, 2) # pred mesh needs to have point_data "error" field to plot color !!!
# plotter.add_text(f"Reconstructed mesh: error", font_size=12)
# plotter.add_mesh(mesh_pred, scalars="error", cmap="jet_r", show_scalar_bar=True,
#     scalar_bar_args=dict(
#         title="",
#         vertical=True,                 
#         title_font_size=16,
#         label_font_size=16,
#         n_labels=5,
#         fmt="%.2f",
#         position_x=0.85,               # INSIDE the subplot
#         position_y=0.1,
#         width=0.05,
#         height=0.7,
#     ),
# )
# plotter.link_views()
# plotter.show(interactive=True)
# camera_pos = plotter.camera_position
# # camera_pos =[(-131401.76942276664, -136192.04038852765, -129296.93927675663),
# #  (-16119.79591370777, -2873.7456654505454, -12539.941061909067),
# #  (0.3081236617252097, 0.46192422080171147, -0.8316765136288149)]
# print(plotter.camera_position)
# plotter.close()

# # off screen render + screenshot with transparent background
# plotter = pv.Plotter(shape=(1, 3), window_size=[1920, 720], off_screen=True)
# plotter.subplot(0, 0)
# plotter.add_text(f"Original mesh: patient {patient_name}", font_size=12)
# plotter.add_mesh(mesh_orig, color="lightgray", opacity=1.0)
# plotter.camera_position = camera_pos
# plotter.background_color = None
# plotter.subplot(0, 1)
# plotter.add_text(f"Reconstructed mesh: patient {patient_name}", font_size=12)
# plotter.add_mesh(mesh_pred, color="lightgray", opacity=1.0)
# plotter.camera_position = camera_pos
# plotter.background_color = None
# plotter.subplot(0, 2) # pred mesh needs to have point_data "error" field to plot color !!!
# plotter.add_text(f"Reconstructed mesh: error", font_size=12)
# plotter.add_mesh(mesh_pred, scalars="error", cmap="jet_r", show_scalar_bar=True,
#     scalar_bar_args=dict(
#         title="",
#         vertical=True,                 
#         title_font_size=16,
#         label_font_size=16,
#         n_labels=5,
#         fmt="%.2f",
#         position_x=0.85,               # INSIDE the subplot
#         position_y=0.1,
#         width=0.05,
#         height=0.7,
#     ),
# )
# plotter.camera_position = camera_pos
# plotter.background_color = None
# plotter.link_views()
# plotter.show(auto_close=False)
# opt = "all"
# plotter.screenshot(f"{patient_name}-pred-vs-gt-{organ}-{version}-reconstructed-from-{opt}.png", transparent_background=True)
# plotter.close()


# # ===== all three surfaces ====== #
# version = "version_114"
# patient_name = "AF069"
# mesh_file = RECONSTRUCTED_MESHES_DIR / f"reconstructed_{patient_name}_{version}_epicardium-res=128.vtp" 
# epi_pred = pv.read(mesh_file)
# mesh_file = RECONSTRUCTED_MESHES_DIR / f"reconstructed_{patient_name}_{version}_ra_endo-res=128.vtp" 
# ra_pred = pv.read(mesh_file)
# mesh_file = RECONSTRUCTED_MESHES_DIR / f"reconstructed_{patient_name}_{version}_la_endo-res=128.vtp" 
# la_pred = pv.read(mesh_file)


# # try interactively
# plotter = pv.Plotter(window_size=[2000, 2000])
# plotter.add_mesh(epi_pred, color = (255, 239, 186),  opacity = 0.3, show_edges=False) #, line_width=0.01)
# plotter.add_mesh(la_pred, color = (255,151,103),  opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.add_mesh(ra_pred, color = "skyblue",  opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.show(interactive=True)
# camera_pos = plotter.camera_position
# print(plotter.camera_position)
# plotter.close()

# # save screenshot with transparent background
# plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)
# plotter.add_mesh(epi_pred, color = (255, 239, 186),  opacity = 0.3, show_edges=False) #, line_width=0.01)
# plotter.add_mesh(la_pred, color = (255,151,103),  opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.add_mesh(ra_pred, color = "skyblue",  opacity = 1.0, show_edges=False) #, line_width=0.01)
# plotter.camera_position = camera_pos
# plotter.background_color = None
# plotter.show(auto_close=False)
# plotter.screenshot(f"{patient_name}-epi-la-ra-reconstructed-{version}.png", transparent_background=True)
# plotter.close()




# # =========================================================== #
# # PLOT SDF ON GRID
# # =========================================================== #
# import pyvista as pv
# import numpy as np
# from scipy.interpolate import RegularGridInterpolator

# box_lim = 105
# resolution = 128

# x = np.linspace(-box_lim, box_lim, resolution)
# y = np.linspace(-box_lim, box_lim, resolution)
# z = np.linspace(-box_lim, box_lim, resolution)
# xx, yy, zz = np.meshgrid(x, y, z)
# nx, ny, nz = xx.shape

# sdf_pred = np.load(f"sdf_epi_AF009_P2R_res={resolution}.npy")

# interp_func = RegularGridInterpolator(
#     (x, y, z),
#     sdf_pred.reshape((resolution, resolution, resolution)),
#     bounds_error=False,  # allow extrapolation if needed
#     fill_value=None      # use NaN outside original grid
# )

# res_new = 4  # new resolution
# x_new = np.linspace(-box_lim, box_lim, res_new)
# y_new = np.linspace(-box_lim, box_lim, res_new)
# z_new = np.linspace(-box_lim, box_lim, res_new)

# xx_new, yy_new, zz_new = np.meshgrid(x_new, y_new, z_new, indexing='ij')
# points_new = np.c_[xx_new.ravel(), yy_new.ravel(), zz_new.ravel()]

# sdf_new = interp_func(points_new)
# sdf_new = sdf_new.reshape((res_new, res_new, res_new))


# xx = xx_new
# yy = yy_new
# zz = zz_new
# sdf_pred = sdf_new


# # ---- Point cloud (for colored vertices) ----
# points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
# pc = pv.PolyData(points)
# pc["sdf"] = sdf_pred.ravel()


# # ---- Build lines for grid ----
# lines = []

# nx, ny, nz = xx.shape

# def idx(i,j,k):
#     return i*ny*nz + j*nz + k

# # Lines along x
# for i in range(nx):
#     for j in range(ny):
#         for k in range(nz-1):
#             lines.append([2, idx(i,j,k), idx(i,j,k+1)])

# # Lines along y
# for i in range(nx):
#     for j in range(ny-1):
#         for k in range(nz):
#             lines.append([2, idx(i,j,k), idx(i,j+1,k)])

# # Lines along z
# for i in range(nx-1):
#     for j in range(ny):
#         for k in range(nz):
#             lines.append([2, idx(i,j,k), idx(i+1,j,k)])

# lines = np.hstack(lines)

# grid_lines = pv.PolyData()
# grid_lines.points = points
# grid_lines.lines = lines

# # mesh = pv.read("reconstructed_AF009_P2R_version_82_epicardium-res=32.vtp")

# # Plot
# plotter = pv.Plotter(window_size=[1620,1620], off_screen=True)
# plotter.add_mesh(grid_lines, color="black", line_width=0.5)
# # plotter.add_mesh(mesh, color='white', opacity=1.0)#show_edges=True, line_width=1.0)
# plotter.add_mesh(pc, scalars="sdf", cmap="jet_r", point_size=25, render_points_as_spheres=True)
# plotter.camera_position = [(-589.9772304724305, 318.86746393583076, 862.9534766439108),
#  (59.6353621951196, -150.2753189897191, -89.67895496790976),
#  (0.22759665313938124, 0.9261058940857274, -0.3008781089069586)]
# plotter.camera_position = [(-279.21074381890026, 170.0030838134723, 439.73175847291566),
#  (87.47862937657513, -94.81578662087888, -98.00441409569488),
#  (0.22759665313938124, 0.9261058940857274, -0.3008781089069586)]
# plotter.background_color = None
# plotter.show(auto_close=False)
# # plotter.screenshot(f"AF009_P2R-sdf_on_grid.png", transparent_background=True)
# plotter.screenshot(f"AF009_P2R-lowres-mesh-marchingcubess.png", transparent_background=True)
# plotter.close()

















# # ======================================================================== # 
# # PLOTTING POINTS AND SDFS

# PATIENTS_NPY_DATA_DIR = Path("/home/davidenava_linux/AtriaProject/deepcsdf_fork/deepcsdf/deepcsdf/deepcsdfatria/data/single_patients_npy")

# patient = "AF009_P2R"

# fname = next( PATIENTS_NPY_DATA_DIR.glob(f"{patient}*"), None)

# data = np.load(fname)

# coords = data[:,:3]
# sdf = data[:,3:]

# eps = 0.005
# near_la = np.where(np.abs(sdf[:,1]) <= eps)
# points = pv.PolyData(coords[near_la])
# # points["sdf_epi"] = sdf[:,0]

# # plotter = pv.Plotter()
# # plotter.add_mesh(points, scalars = "sdf_epi", cmap = "jet_r", render_points_as_spheres=True)
# # plotter.show()

# plotter = pv.Plotter()
# plotter.add_mesh(points)
# plotter.show()






























# # =============================== #
# # PLOTTING LOSSES AND OTHER STUFF
# # =============================== #
# import json
# import matplotlib.pyplot as plt

# loss_data = np.array( json.load(open("train_loss_version_89.json")) )

# print(loss_data.shape)

# loss = loss_data[:,2]
# k = 2
# smoothed = loss.copy()
# for i in range(k, len(loss) - k):
#     s = np.mean( loss[i:int(i+2*k)] )
#     smoothed[k+i] = s

# fig_width = 5
# fig_height = 6.25      # 4:5 aspect (width:height)

# grid_color = (0.7, 0.7, 0.7)
# grid_width = 1.0
# grid_alpha = 0.5

# loss_color = np.array((210, 216, 226)) / 255
# smooth_color = np.array((66, 80, 102)) / 255

# fig, ax = plt.subplots(figsize=(fig_width, fig_height))
# ax.plot(np.arange(len(loss[10:])), loss[10:], c=loss_color, linewidth=2)
# ax.plot(np.arange(len(smoothed[10:])), smoothed[10:], c=smooth_color, linewidth=2)
# ax.set_xticks(np.linspace(0, len(loss[10:]), 5))
# ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
# ax.grid(
#     True,
#     linestyle='-',
#     linewidth=grid_width,
#     color=grid_color,
#     alpha=grid_alpha
# )
# ax.set_axisbelow(True)
# plt.savefig(
#     "loss_plot.svg",
#     format="svg",
#     transparent=True,
#     bbox_inches="tight"
# )
# plt.show()