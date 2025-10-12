from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
import re
import uuid
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.manufactoria_parser import create_robot_factory, ParseError
app = Flask(__name__)

# Problems directory path
PROBLEMS_DIR = 'problems/difficulty'

# Global cache for all problems
_problems_cache = None

def invalidate_problems_cache():
    """Invalidate the problems cache when problems are modified."""
    global _problems_cache
    _problems_cache = None

def get_problem_files():
    """Get all available problem files from the problems directory."""
    files = []
    if os.path.exists(PROBLEMS_DIR):
        for f in os.listdir(PROBLEMS_DIR):
            if f.endswith('.jsonl'):
                files.append(f)
    return files

def get_problem_types():
    """Get all available problem types from filenames in the problems directory."""
    types = set()
    if os.path.exists(PROBLEMS_DIR):
        for f in os.listdir(PROBLEMS_DIR):
            if f.endswith('.jsonl'):
                # Extract problem type from filename (remove .jsonl and last two parts: color_mode and difficulty)
                filename_without_ext = f.replace('.jsonl', '')
                parts = filename_without_ext.split('_')
                if len(parts) >= 3:  # Ensure we have at least problem_type + color_mode + difficulty
                    problem_type = '_'.join(parts[:-2])  # Join all parts except the last two
                    types.add(problem_type)
    return sorted(list(types))

def load_problems_from_file(filename):
    """Load problems from a specific JSONL file. Filename can include subdirectory (e.g., 'c2/starts_with.jsonl')."""
    filepath = os.path.join(PROBLEMS_DIR, filename)
    if not os.path.exists(filepath):
        return []
    
    problems = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
    except (json.JSONDecodeError, FileNotFoundError):
        return []
    
    return problems

def load_all_problems():
    """Load all problems from all files in both c2 and c4 directories."""
    global _problems_cache
    if _problems_cache is not None:
        return _problems_cache

    all_problems = []
    for filename in get_problem_files():
        problems = load_problems_from_file(filename)
        all_problems.extend(problems)
    _problems_cache = all_problems
    return all_problems

def load_problems_by_type(problem_type):
    """Load problems of a specific type from the problems directory."""
    all_problems = []
    if os.path.exists(PROBLEMS_DIR):
        for f in os.listdir(PROBLEMS_DIR):
            if f.endswith('.jsonl') and f.startswith(f"{problem_type}_"):
                problems = load_problems_from_file(f)
                all_problems.extend(problems)
    return all_problems

def save_problems_to_file(problems, filename):
    """Save problems to a specific JSONL file. Filename can include subdirectory."""
    filepath = os.path.join(PROBLEMS_DIR, filename)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        for problem in problems:
            f.write(json.dumps(problem) + '\n')

def save_problem(problem):
    """Save a single problem to the appropriate file based on its problem_type, color_mode, and difficulty_level."""
    problem_type = problem.get('problem_type', 'misc')
    color_mode = problem.get('color_mode', 'two_color')
    difficulty_level = problem.get('difficulty_level', 'basic')

    # Create filename with embedded color and difficulty info
    filename = f"{problem_type}_{color_mode}_{difficulty_level}.jsonl"

    # Load existing problems of this type, color mode, and difficulty level
    existing_problems = load_problems_from_file(filename)

    # Find and update if problem exists, otherwise add new
    problem_updated = False
    for i, existing_problem in enumerate(existing_problems):
        if existing_problem['id'] == problem['id']:
            existing_problems[i] = problem
            problem_updated = True
            break

    if not problem_updated:
        existing_problems.append(problem)

    # Save back to file
    save_problems_to_file(existing_problems, filename)
    invalidate_problems_cache() # Invalidate cache after saving

def delete_problem_by_id(problem_id):
    """Delete a problem by ID from all problem files."""
    for filename in get_problem_files():
        problems = load_problems_from_file(filename)
        original_count = len(problems)
        problems = [p for p in problems if p['id'] != problem_id]
        
        if len(problems) < original_count:
            save_problems_to_file(problems, filename)
            invalidate_problems_cache() # Invalidate cache after deleting
            return True
    return False

def generate_new_id():
    """Generate a new UUID for a new problem."""
    return str(uuid.uuid4())

@app.route('/')
def index():
    """Main page showing list of problems with filtering by type, color mode, and pagination."""
    problem_type = request.args.get('type', '')
    color_mode = request.args.get('color_mode', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 12))  # Default 12 problems per page
    
    # Ensure page is at least 1
    page = max(1, page)
    # Ensure per_page is reasonable (between 6 and 50)
    per_page = max(6, min(50, per_page))
    
    # Load problems based on filters
    if problem_type:
        all_problems = load_problems_by_type(problem_type)
    else:
        all_problems = load_all_problems()
    
    # Apply color mode filter if specified
    if color_mode:
        all_problems = [p for p in all_problems if p.get('color_mode', 'two_color') == color_mode]
    
    # Calculate pagination
    total_problems = len(all_problems)
    total_pages = (total_problems + per_page - 1) // per_page  # Ceiling division
    
    # Ensure current page doesn't exceed total pages
    if page > total_pages and total_pages > 0:
        page = total_pages
    
    # Calculate start and end indices for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get problems for current page
    problems = all_problems[start_idx:end_idx]
    
    # Create pagination info
    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total_problems,
        'total_pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_num': page - 1 if page > 1 else None,
        'next_num': page + 1 if page < total_pages else None,
        'pages': list(range(max(1, page - 2), min(total_pages + 1, page + 3)))  # Show 5 pages around current
    }
    
    problem_types = get_problem_types()
    return render_template('index.html', 
                         problems=problems, 
                         problem_types=problem_types, 
                         selected_type=problem_type,
                         selected_color_mode=color_mode,
                         pagination=pagination)

@app.route('/problem/<string:problem_id>')
def view_problem(problem_id):
    """View a specific problem."""
    all_problems = load_all_problems()
    problem = next((p for p in all_problems if p['id'] == problem_id), None)
    if not problem:
        return redirect(url_for('index'))
    return render_template('problem.html', problem=problem)

@app.route('/edit/<string:problem_id>')
def edit_problem(problem_id):
    """Edit a specific problem."""
    all_problems = load_all_problems()
    problem = next((p for p in all_problems if p['id'] == problem_id), None)
    if not problem:
        return redirect(url_for('index'))
    
    problem_types = get_problem_types()
    return render_template('edit.html', problem=problem, problem_types=problem_types)

@app.route('/new')
def new_problem():
    """Create a new problem."""
    problem_types = get_problem_types()
    return render_template('edit.html', problem=None, problem_types=problem_types)

@app.route('/api/problems', methods=['GET'])
def api_get_problems():
    """API endpoint to get problems, optionally filtered by type."""
    problem_type = request.args.get('type', '')
    
    if problem_type:
        problems = load_problems_by_type(problem_type)
    else:
        problems = load_all_problems()
    
    return jsonify(problems)

@app.route('/api/problem_types', methods=['GET'])
def api_get_problem_types():
    """API endpoint to get all available problem types."""
    return jsonify(get_problem_types())

@app.route('/api/problems', methods=['POST'])
def api_create_problem():
    """API endpoint to create a new problem."""
    data = request.json
    
    # Calculate next index for this problem type, color mode, and difficulty level
    problem_type = data.get('problem_type', 'misc')
    color_mode = data.get('color_mode', 'two_color')
    difficulty_level = data.get('difficulty_level', 'basic')
    filename = f"{problem_type}_{color_mode}_{difficulty_level}.jsonl"
    existing_problems = load_problems_from_file(filename)
    next_index = len(existing_problems)
    
    new_problem = {
        'id': generate_new_id(),
        'index': next_index,
        'name': data.get('name', ''),
        'description': data.get('description', ''),
        'criteria': data.get('criteria', ''),
        'difficulty': data.get('difficulty', 'basic'),
        'problem_type': problem_type,
        'color_mode': color_mode,
        'difficulty_level': difficulty_level,
        'available_nodes': data.get('available_nodes', [
            'START', 'END', 'PULLER_RB', 'PULLER_YG',
            'PAINTER_RED', 'PAINTER_BLUE', 'PAINTER_YELLOW', 'PAINTER_GREEN'
        ]),
        'test_cases': data.get('test_cases', []),
        'solutions': data.get('solutions', []),
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    save_problem(new_problem)
    return jsonify(new_problem), 201

@app.route('/api/problems/<string:problem_id>', methods=['PUT'])
def api_update_problem(problem_id):
    """API endpoint to update a problem."""
    data = request.json
    all_problems = load_all_problems()
    
    problem = next((p for p in all_problems if p['id'] == problem_id), None)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    
    # Check if problem_type, color_mode, or difficulty_level is changing
    old_problem_type = problem.get('problem_type', 'misc')
    old_color_mode = problem.get('color_mode', 'two_color')
    old_difficulty_level = problem.get('difficulty_level', 'basic')
    new_problem_type = data.get('problem_type', old_problem_type)
    new_color_mode = data.get('color_mode', old_color_mode)
    new_difficulty_level = data.get('difficulty_level', old_difficulty_level)

    problem.update({
        'name': data.get('name', problem['name']),
        'description': data.get('description', problem['description']),
        'criteria': data.get('criteria', problem['criteria']),
        'difficulty': data.get('difficulty', problem.get('difficulty', 'basic')),
        'problem_type': new_problem_type,
        'color_mode': new_color_mode,
        'difficulty_level': new_difficulty_level,
        'available_nodes': data.get('available_nodes', problem['available_nodes']),
        'test_cases': data.get('test_cases', problem['test_cases']),
        'solutions': data.get('solutions', problem['solutions']),
        'updated_at': datetime.now().isoformat()
    })

    # If problem type, color mode, or difficulty level changed, remove from old file
    if old_problem_type != new_problem_type or old_color_mode != new_color_mode or old_difficulty_level != new_difficulty_level:
        # Remove from old file
        old_filename = f"{old_problem_type}_{old_color_mode}_{old_difficulty_level}.jsonl"
        old_problems = load_problems_from_file(old_filename)
        old_problems = [p for p in old_problems if p['id'] != problem_id]
        save_problems_to_file(old_problems, old_filename)
    
    # Save to appropriate file
    save_problem(problem)
    return jsonify(problem)

@app.route('/api/problems/<string:problem_id>', methods=['DELETE'])
def api_delete_problem(problem_id):
    """API endpoint to delete a problem."""
    success = delete_problem_by_id(problem_id)
    return jsonify({'success': success})

@app.route('/api/validate_dsl', methods=['POST'])
def api_validate_dsl():
    """API endpoint to validate DSL syntax."""
    data = request.json
    dsl_code = data.get('dsl', '')
    
    try:
        factory = create_robot_factory(dsl_code)
        return jsonify({'valid': True, 'message': 'DSL is valid'})
    except ParseError as e:
        return jsonify({'valid': False, 'message': str(e)})
    except Exception as e:
        return jsonify({'valid': False, 'message': f'Unexpected error: {str(e)}'})

@app.route('/api/test_solution', methods=['POST'])
def api_test_solution():
    """API endpoint to test a solution against test cases."""
    data = request.json
    dsl_code = data.get('dsl', '')
    test_cases = data.get('test_cases', [])
    
    try:
        factory = create_robot_factory(dsl_code)
        results = []
        
        for test_case in test_cases:
            input_tape = test_case.get('input', '')
            expected_output = test_case.get('expected_output', '')
            expected_accepted = test_case.get('expected_accepted', True)
            check_output = test_case.get('check_output', True)
            
            result = factory.process_robot(input_tape)
            
            if check_output:
                # Check if expected_output contains regex patterns
                has_regex_patterns = any(char in expected_output for char in ['.', '+', '*', '?', '|', '(', ')'])
                
                if has_regex_patterns:
                    # Use regex matching
                    try:
                        output_matches = bool(re.fullmatch(expected_output, result.final_tape))
                    except re.error:
                        # If regex pattern is invalid, fall back to exact matching
                        output_matches = result.final_tape == expected_output
                else:
                    # Use exact matching
                    output_matches = result.final_tape == expected_output
                
                passed = (output_matches and result.finished) == expected_accepted
            else:
                passed = (result.finished == expected_accepted)
            
            test_result = {
                'input': input_tape,
                'expected_output': expected_output,
                'actual_output': result.final_tape,
                'expected_accepted': expected_accepted,
                'actual_accepted': result.finished,
                'check_output': check_output,
                'passed': passed,
                'path': result.path,
                'rejection_reason': result.rejection_reason
            }
            results.append(test_result)
        
        all_passed = all(r['passed'] for r in results)
        
        return jsonify({
            'valid': True,
            'all_passed': all_passed,
            'results': results
        })
        
    except ParseError as e:
        return jsonify({'valid': False, 'message': str(e)})
    except Exception as e:
        return jsonify({'valid': False, 'message': f'Unexpected error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)