#!/bin/bash
# Script to generate datasets for each problem family separately
export OPENAI_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Array of all problem families
families=(
    "relational_joining"
    "aggregation_and_having" 
    "set_algebra"
    "subquery_semantics"
    "window_analytics"
    "hierarchical_paths"
    "data_mutation"
)

# Base config template
config_template="config.yaml"

echo "🚀 Generating datasets for all problem families separately..."

for family in "${families[@]}"; do
    echo ""
    echo "📊 Processing family: $family"
    
    # Create temporary config with single family
    temp_config="config_${family}.yaml"
    
    # Copy base config and modify the single_problem_family setting
    sed "s/single_problem_family: null/single_problem_family: \"$family\"/" "$config_template" > "$temp_config"
    
    # Create output directory for this family
    output_dir="result_${family}"
    
    # Update the result path in temp config
    sed -i.bak "s|result_base: result|result_base: $output_dir|" "$temp_config"
    
    echo "  📁 Output directory: $output_dir"
    echo "  ⚙️  Config file: $temp_config"
    echo "  🏃 Running pipeline..."
    
    # Run the pipeline
    python run.py --config "$temp_config"
    
    # Clean up temporary config
    rm "$temp_config" "$temp_config.bak" 2>/dev/null
    
    echo "  ✅ Completed: $family"
done

echo ""
echo "🎉 All problem families generated successfully!"
echo ""
echo "📊 Results summary:"
for family in "${families[@]}"; do
    output_dir="result_${family}"
    if [ -f "$output_dir/dataset.jsonl" ]; then
        count=$(wc -l < "$output_dir/dataset.jsonl" 2>/dev/null || echo "0")
        echo "  $family: $count samples in $output_dir/dataset.jsonl"
    else
        echo "  $family: No dataset.jsonl found"
    fi
done
