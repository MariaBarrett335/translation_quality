#!/bin/bash
#SBATCH --job-name=convert_sft
#SBATCH --output=.logs/%j.out # Name of stdout output file
#SBATCH --error=.logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --time=01:00:00
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --account=project_462000353  # Project for billing

# Usage: sbatch convert_to_sft_format.sh input_file.jsonl output_directory [input_column] [output_column] [instruction]

# Get command line arguments
INPUT_FILE=$1
OUTPUT_DIR=$2
INPUT_COLUMN=${3:-"sentence"}
OUTPUT_COLUMN=${4:-"fluency_score"}
INSTRUCTION=${5:-'Rate the fluency of this Danish sentence on a scale of 1-5 where 1 means either not Danish or very dysfluent and 5 means indistinguishable from text written by a native speaker: "%s"'}

export INPUT_FILE
export OUTPUT_DIR  
export INPUT_COLUMN
export OUTPUT_COLUMN
export INSTRUCTION

# Validate required arguments
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch convert_to_sft_format.sh input_file.jsonl output_directory [input_column] [output_column] [instruction]"
    echo ""
    echo "Arguments:"
    echo "  input_file.jsonl  - Input JSONL file"
    echo "  output_directory  - Output directory for train.jsonl and test.jsonl"
    echo "  input_column      - Column name for input text (default: 'sentence')"
    echo "  output_column     - Column name for target output (default: 'fluency_score')"
    echo "  instruction       - User instruction template with %s placeholder (default: fluency rating prompt)"
    echo ""
    echo "Examples:"
    echo "  # Basic fluency rating (default)"
    echo "  sbatch convert_to_sft_format.sh data.jsonl output_dir"
    echo ""
    echo "  # Custom columns"
    echo "  sbatch convert_to_sft_format.sh data.jsonl output_dir text score"
    echo ""
    echo "  # Custom instruction"
    echo "  sbatch convert_to_sft_format.sh data.jsonl output_dir sentence rating \"Evaluate this text: %s\""
    echo ""
    echo "  # Translation task"
    echo "  sbatch convert_to_sft_format.sh data.jsonl output_dir english danish \"Translate to Danish: %s\""
    exit 1
fi

echo "Starting SFT format conversion..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Input column: $INPUT_COLUMN"
echo "Output column: $OUTPUT_COLUMN"
echo "Instruction template: $INSTRUCTION"
echo ""

# Run the Python conversion script
python3 << EOF
import json
import os
import sys
from datetime import datetime

def convert_to_sft_format(input_file, output_dir, input_col, output_col, instruction_template, train_split=0.8):
    """
    Convert JSONL to SFT format with configurable columns and instruction
    """
    print("=" * 60)
    print("CONVERTING TO SFT FORMAT")
    print("=" * 60)
    print(f"Input column: '{input_col}'")
    print(f"Output column: '{output_col}'")
    print(f"Instruction: '{instruction_template}'")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Read input data
    print(f"📖 Reading data from: {input_file}")
    data = []
    errors = 0
    missing_input_col = 0
    missing_output_col = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    
                    # Check if required columns exist
                    if input_col not in item:
                        missing_input_col += 1
                        if missing_input_col <= 5:  # Show first 5 errors
                            print(f"⚠️  Line {line_num}: Missing input column '{input_col}'")
                        continue
                        
                    if output_col not in item:
                        missing_output_col += 1
                        if missing_output_col <= 5:  # Show first 5 errors
                            print(f"⚠️  Line {line_num}: Missing output column '{output_col}'")
                        continue
                    
                    # Check if input column has content
                    input_value = item[input_col]
                    if not input_value or (isinstance(input_value, str) and not input_value.strip()):
                        errors += 1
                        continue
                    
                    # Check if output column has valid value
                    output_value = item[output_col]
                    if output_value is None:
                        errors += 1
                        continue
                    
                    data.append(item)
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON error on line {line_num}: {e}")
                    errors += 1
                except Exception as e:
                    print(f"❌ Error processing line {line_num}: {e}")
                    errors += 1
                    
                if line_num % 5000 == 0:
                    print(f"   Processed {line_num} lines...")
                    
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        sys.exit(1)
    
    print(f"✅ Successfully loaded {len(data)} valid examples")
    if errors > 0:
        print(f"⚠️  Skipped {errors} entries with missing/invalid data")
    if missing_input_col > 0:
        print(f"⚠️  {missing_input_col} entries missing input column '{input_col}'")
    if missing_output_col > 0:
        print(f"⚠️  {missing_output_col} entries missing output column '{output_col}'")
    
    if len(data) == 0:
        print("❌ No valid data found. Exiting.")
        sys.exit(1)
    
    # Show data distribution for output column
    output_values = [str(item[output_col]) for item in data]
    unique_outputs = list(set(output_values))
    print(f"📊 Output value distribution:")
    for val in sorted(unique_outputs):
        count = output_values.count(val)
        percentage = count / len(data) * 100
        print(f"   '{val}': {count} ({percentage:.1f}%)")
    
    # Convert to ChatML format
    print(f"🔄 Converting to ChatML format...")
    converted_data = []
    
    for i, item in enumerate(data):
        try:
            input_text = str(item[input_col]).strip()
            output_text = str(item[output_col]).strip()
            
            # Format the instruction with the input text
            # Handle both %s and {} style formatting
            try:
                if '%s' in instruction_template:
                    user_content = instruction_template % input_text
                elif '{}' in instruction_template:
                    user_content = instruction_template.format(input_text)
                else:
                    # No placeholder, append input text
                    user_content = f"{instruction_template} {input_text}"
            except (TypeError, ValueError) as e:
                print(f"⚠️  Warning: Instruction formatting error for item {i+1}: {e}")
                user_content = f"{instruction_template} {input_text}"
            
            sft_entry = {
                'messages': [
                    {
                        'role': 'user',
                        'content': user_content
                    },
                    {
                        'role': 'assistant', 
                        'content': output_text
                    }
                ]
            }
            converted_data.append(sft_entry)
            
        except Exception as e:
            print(f"⚠️  Warning: Error converting item {i+1}: {e}")
            continue
    
    print(f"✅ Converted {len(converted_data)} examples to ChatML format")
    
    # Create train/test split
    print(f"📊 Creating train/test split ({train_split*100:.0f}%/{(1-train_split)*100:.0f}%)...")
    split_idx = int(train_split * len(converted_data))
    train_data = converted_data[:split_idx]
    test_data = converted_data[split_idx:]
    
    print(f"   Train examples: {len(train_data)}")
    print(f"   Test examples: {len(test_data)}")
    
    # Create output directory
    print(f"📁 Creating output directory: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        sys.exit(1)
    
    # Write train split
    train_file = os.path.join(output_dir, 'train.jsonl')
    print(f"💾 Writing training data to: {train_file}")
    try:
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"❌ Error writing train file: {e}")
        sys.exit(1)
    
    # Write test split
    test_file = os.path.join(output_dir, 'test.jsonl')
    print(f"💾 Writing test data to: {test_file}")
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"❌ Error writing test file: {e}")
        sys.exit(1)
    
    # Verification
    print(f"🔍 Verifying output files...")
    try:
        train_lines = sum(1 for _ in open(train_file, 'r'))
        test_lines = sum(1 for _ in open(test_file, 'r'))
        
        print(f"   Train file: {train_lines} lines")
        print(f"   Test file: {test_lines} lines")
        
        if train_lines == len(train_data) and test_lines == len(test_data):
            print("✅ Verification passed!")
        else:
            print("⚠️  Warning: Line count mismatch")
            
    except Exception as e:
        print(f"⚠️  Could not verify files: {e}")
    
    # Show example
    print(f"\n📝 Example ChatML format:")
    if train_data:
        example = train_data[0]
        print(json.dumps(example, ensure_ascii=False, indent=2))
    
    print(f"\n✅ Conversion completed successfully!")
    print(f"📂 Output directory: {output_dir}")
    print(f"📁 Files created:")
    print(f"   - train.jsonl ({len(train_data)} examples)")
    print(f"   - test.jsonl ({len(test_data)} examples)")
    
    return len(converted_data)

# Main execution
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"🚀 Process started at: {start_time}")
    
    input_file = os.environ.get('INPUT_FILE', '$INPUT_FILE')
    output_dir = os.environ.get('OUTPUT_DIR', '$OUTPUT_DIR') 
    input_col = os.environ.get('INPUT_COLUMN', '$INPUT_COLUMN')
    output_col = os.environ.get('OUTPUT_COLUMN', '$OUTPUT_COLUMN')
    instruction = os.environ.get('INSTRUCTION', '$INSTRUCTION')
    
    try:
        total_examples = convert_to_sft_format(input_file, output_dir, input_col, output_col, instruction)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 SUCCESS!")
        print(f"⏰ Completed at: {end_time}")
        print(f"⏱️  Duration: {duration}")
        print(f"📊 Total examples processed: {total_examples}")
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

EOF

echo ""
echo "🏁 Job completed at: $(date)"