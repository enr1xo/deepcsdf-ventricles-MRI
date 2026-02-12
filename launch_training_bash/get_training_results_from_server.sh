# SERVER_PATH=davide.navarri@baldo.disi.unitn.it:/home/davide.navarri/deepcsdfatria/experiments/deepsdf_atria_training/version_
# LOCAL_PATH=/home/davidenava_linux/AtriaProject/deepcsdf_fork/deepcsdf/deepcsdf/deepcsdfatria/experiments/deepsdf_atria_training
# rsync -avz --progress  --exclude="*.yaml" --exclude="events.*" --exclude="checkpoints/" $SERVER_PATH" "$LOCAL_PATH"

# rsync -avz \
#     --include="*/" \
#     --include="file1" \
#     --include="*<region>_pippobaudo.vtp" \
#     --exclude="*" \
#     src/ davide.navarri@baldo.disi.unitn.it:/home/davide.navarri/deepcsdf_atria/
# rsync -avz --include="*/" /home/davidenava_linux/AtriaProject/deepcsdf_fork/deepcsdf/deepcsdf/deepcsdfatria/data/single_patients_npy davide.navarri@baldo.disi.unitn.it:/home/davide.navarri/deepcsdf_atria/data/


rsync -avz --progress  --exclude="*.yaml" --exclude="events.*" --exclude="checkpoints/" $SERVER_PATH" "$LOCAL_PATH"

rsync -avz --progress  --exclude="*.yaml" --exclude="events.*" --exclude="checkpoints/" davide.navarri@baldo.disi.unitn.it:/home/davide.navarri/deepcsdfatria/experiments/deepsdf_atria_training/version_30 /home/davidenava_linux/AtriaProject/deepcsdf_fork/deepcsdf/deepcsdf/deepcsdfatria/experiments/deepsdf_atria_training 

rsync -avz --progress  --exclude="*.yaml" --exclude="events.*" --exclude="checkpoints/" davide.navarri@baldo.disi.unitn.it:/home/davide.navarri/deepcsdfatria/experiments/deepsdf_atria_training_concurrent/ /home/davidenava_linux/AtriaProject/deepcsdf_fork/deepcsdf/deepcsdf/deepcsdfatria/experiments/deepsdf_atria_training_concurrent