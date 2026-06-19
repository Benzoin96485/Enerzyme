INPUT_CSV=COMT.csv
QP_OUTPUT_DIR=./qp
CONFIG_YAML=qp_config.yaml
cp qp_config_template.yaml $CONFIG_YAML
sed -i "s|__INPUT_CSV__|$INPUT_CSV|g" $CONFIG_YAML
sed -i "s|__OUTPUT_DIR__|$QP_OUTPUT_DIR|g" $CONFIG_YAML
mkdir -p $QP_OUTPUT_DIR
python gen_cluster.py -m ../master_list.csv -i $INPUT_CSV -q $QP_OUTPUT_DIR -c $CONFIG_YAML -o .