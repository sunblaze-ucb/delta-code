
# Explorative Setting 


## Setting 1 -- contains patterns

### Train 
manufactoria/basic_mix_train 1.0
manufactoria/contains_mix_train 1.0

### Test
manufactoria/contains_ood_mix_test 50
manufactoria/contains_mix_test 50

## Setting 2 -- numerical comparison

### Train
manufactoria/basic_mix_train 1.0
manufactoria/numerical_comparison_train 1.0

### Test
manufactoria/numerical_comparison_test 50
manufactoria/numerical_comparison_two_color_hard 50


## Setting 3 -- regex pattern

### Train
manufactoria/basic_mix_train 1.0
manufactoria/regex_pattern_train 1.0

### Test
manufactoria/regex_pattern_test 50
manufactoria/regex_pattern_four_color_hard 50



# Compositional Setting 

## Setting 1 -- replace pattern 

### Train
manufactoria/basic_mix_train 1.0
manufactoria/regex_pattern_train 1.0
manufactoria/prepend_sequence_train 1.0

### Test
manufactoria/regex_pattern_test 50
manufactoria/prepend_sequence_test 50
manufactoria/prepend_sequence_two_color_hard 50

## Setting 2 -- numerical operation

### Train
manufactoria/basic_mix_train 1.0
manufactoria/numerical_comparison_train 1.0
manufactoria/prepend_sequence_train 1.0

### Test
manufactoria/numerical_comparison_test 50
manufactoria/prepend_sequence_test 50
manufactoria/numerical_operations_two_color_easy 7
manufactoria/numerical_operations_two_color_hard 14
manufactoria/numerical_operations_two_color_medium 21
manufactoria/numerical_max_min_two_color_medium 50     


# Transformative Setting 

## Setting 1 -- start_with -> end_with

### Train 
manufactoria/manufactoria_basic_mix_train 1.0 

### Test
manufactoria/ends_with_two_color_medium 50


## Setting 2 -- regex match

### Train
manufactoria/basic_mix_train 1.0
manufactoria/regex_pattern_train 1.0

### Test
manufactoria/regex_same_num_four_color_hard 50


## (not consider) Setting 3 -- replace sequence (one sequence to two sequence)

### Train
manufactoria/basic_mix_train 1.0
manufactoria/prepend_sequence_train 1.0

### Test
manufactoria/prepend_sequence_test 50
manufactoria/prepend_sequence_two_color_hard 50


# Step-wise 

regex_pattern
numerical_comparison
prepend_sequence


# Joint training

# Setting 0 -- Basic  (basic, transformative)

### Train 
manufactoria/manufactoria_basic_mix_train 1.0 

### Test
manufactoria/ends_with_two_color_medium 50
manufactoria/contains_mix_test 50
manufactoria/numerical_comparison_test 50
manufactoria/regex_pattern_test 50


## Setting 1 -- contains patterns  (explorative)

### Train 
manufactoria/basic_mix_train 1.0
manufactoria/contains_mix_train 1.0

### Test
manufactoria/contains_ood_mix_test 50
manufactoria/contains_mix_test 50



## Setting 2 -- numerical comparison (explorative, compositional)  (step-wise)

### Train
manufactoria/basic_mix_train 1.0
manufactoria/numerical_comparison_train 1.0
manufactoria/prepend_sequence_train 1.0

### Test
manufactoria/numerical_comparison_test 50
manufactoria/prepend_sequence_test 50
manufactoria/numerical_comparison_two_color_hard 50   (expl)
manufactoria/numerical_operations_two_color_easy 7    (comp)
manufactoria/numerical_operations_two_color_hard 14   (comp)
manufactoria/numerical_operations_two_color_medium 21 (comp)
manufactoria/numerical_max_min_two_color_medium 50    (comp)


## Setting 3 -- regex pattern  (explorative, compositional, transformative)  (step-wise)

### Train
manufactoria/basic_mix_train 1.0
manufactoria/regex_pattern_train 1.0
manufactoria/prepend_sequence_train 1.0

### Test
manufactoria/regex_pattern_test 50
manufactoria/prepend_sequence_test 50
manufactoria/regex_pattern_four_color_hard 50        (expl)
manufactoria/prepend_sequence_two_color_hard 50      (comp)
manufactoria/regex_same_num_four_color_hard 50       (trans)

