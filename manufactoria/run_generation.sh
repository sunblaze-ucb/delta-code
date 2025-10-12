python problem_difficulty_generator.py

python ensemble_generator.py --config configs/basic_mix.yaml
python ensemble_generator.py --config configs/compr.yaml

python ensemble_generator.py --config configs/ends.yaml
python ensemble_generator.py --config configs/regex.yaml
python ensemble_generator.py --config configs/has.yaml

python ensemble_generator.py --config configs/prepend.yaml
python ensemble_generator.py --config configs/mutate.yaml
python ensemble_generator.py --config configs/bit_op.yaml

python ensemble_generator.py --config configs/fdiv.yaml
python ensemble_generator.py --config configs/symm.yaml
python ensemble_generator.py --config configs/minmax.yaml
python ensemble_generator.py --config configs/add.yaml



python hf_file_wrapper.py --input problems/ensembles/ --output problems/wrapped/ --upload false



