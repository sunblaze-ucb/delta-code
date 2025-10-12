

for letter in A B C D F G; do
  python generate.py \
      --types "$letter" \
      --difficulties 0 1 2 3 4 \
      --scenes-per-config 5 \
      --dataset-name scene_$letter \
      --timestamps-per-scene 15 \
      --separate-by-difficulty \
      --prompt-setting function \
      --skip-existing
done

case "$letter" in
    A) name=ROTOBJ ;;
    B) name=ROTBOX ;;
    C) name=MOVBOX ;;
    D) name=GRAVITY ;;
    F) name=MULTIBOX ;;
    G) name=MULTIOBJ ;;
    *) echo "Unknown letter: $letter" >&2; continue ;;
  esac


 for letter in AB AC AD AF AG BC BD BF BG CD CF CG DF DG FG; do

 case "$letter" in
    AB) name=ROTOBJ_ROTBOX ;;
    AC) name=ROTOBJ_MOVBOX ;;
    AD) name=ROTOBJ_GRAVITY ;;
    AF) name=ROTOBJ_MULTIBOX ;;
    AG) name=ROTOBJ_MULTIOBJ ;;
    BC) name=ROTBOX_MOVBOX ;;
    BD) name=ROTBOX_GRAVITY ;;
    BF) name=ROTBOX_MULTIBOX ;;
    BG) name=ROTBOX_MULTIOBJ ;;
    CD) name=MOVBOX_GRAVITY ;;
    CF) name=MOVBOX_MULTIBOX ;;
    CG) name=MOVBOX_MULTIOBJ ;;
    DF) name=MULTIBOX_GRAVITY ;;
    DG) name=MULTIBOX_MULTIOBJ ;;
    FG) name=MULTIOBJ_MULTIOBJ ;;
  esac
  for level in basic easy medium hard; do
  python step_2_upload_to_huggingface.py \
    --action dataset \
    --org ballsim \
    --repo-name "ballsim-${letter}-${level}" \
    --data-path "data/datasets/dataset_scene_${letter}_${level}.jsonl" \
    --train-test \
    --train-ratio 0.00 \
    --seed 42
  done
  done
