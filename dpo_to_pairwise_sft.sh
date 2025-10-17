#!/bin/bash
#SBATCH --job-name=convert_dpo_to_sft
#SBATCH --output=.logs/%j.out # Name of stdout output file
#SBATCH --error=.logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --time=01:00:00
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1    # 1 task per node
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --account=project_462000353  # Project for billing

# Usage: sbatch convert_dpo_to_sft.sh input_file.jsonl output_file.jsonl [--no-ab-conversion]

# Get command line arguments
INPUT_FILE=$1
OUTPUT_FILE=$2
NO_AB_CONVERSION=$3

export INPUT_FILE
export OUTPUT_FILE
export NO_AB_CONVERSION

# Validate required arguments
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch convert_dpo_to_sft.sh input_file.jsonl output_file.jsonl [--no-ab-conversion]"
    echo ""
    echo "Arguments:"
    echo "  input_file.jsonl   - Input JSONL file in DPO format"
    echo "  output_file.jsonl  - Output JSONL file in SFT format"
    echo "  --no-ab-conversion - Optional: disable automatic conversion of 1/2 responses to A/B format"
    echo ""
    echo "Examples:"
    echo "  # Basic conversion with A/B format enforcement"
    echo "  sbatch convert_dpo_to_sft.sh dpo_data.jsonl sft_data.jsonl"
    echo ""
    echo "  # Conversion without A/B format changes"
    echo "  sbatch convert_dpo_to_sft.sh dpo_data.jsonl sft_data.jsonl --no-ab-conversion"
    exit 1
fi

echo "Starting DPO to SFT format conversion..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "A/B conversion: $([ "$NO_AB_CONVERSION" = "--no-ab-conversion" ] && echo "disabled" || echo "enabled")"
echo ""

# Run the Python conversion script
python3 << 'EOF'
import json
import os
import sys
import re
from datetime import datetime

def ensure_ab_format(content):
    """
    Ensure the assistant's response is in A/B format, not 1/2 format.
    """
    # If the content is just "1", convert to "A"
    if content.strip() == "1":
        return "A"
    # If the content is just "2", convert to "B"
    elif content.strip() == "2":
        return "B"
    # If it's already A or B, leave it as is
    elif content.strip() in ["A", "B"]:
        return content.strip()
    # For any other content, leave unchanged
    else:
        return content

def convert_user_content_to_ab_format(content):
    """
    Convert user content from 1/2 format to A/B format.
    Changes numbered questions and instructions to use letters instead.
    """
    # Convert "1)" and "2)" at the start of lines to "A)" and "B)"
    content = re.sub(r'^(\s*)1\)', r'\1A)', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s*)2\)', r'\1B)', content, flags=re.MULTILINE)
    
    # Convert "1." and "2." at the start of lines to "A." and "B."
    content = re.sub(r'^(\s*)1\.', r'\1A.', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s*)2\.', r'\1B.', content, flags=re.MULTILINE)
    
    # Convert "Option 1" and "Option 2" to "Option A" and "Option B"
    content = re.sub(r'\bOption\s+1\b', 'Option A', content, flags=re.IGNORECASE)
    content = re.sub(r'\bOption\s+2\b', 'Option B', content, flags=re.IGNORECASE)
    
    # Convert "Sentence 1" and "Sentence 2" to "Sentence A" and "Sentence B"
    content = re.sub(r'\bSentence\s+1\b', 'Sentence A', content, flags=re.IGNORECASE)
    content = re.sub(r'\bSentence\s+2\b', 'Sentence B', content, flags=re.IGNORECASE)
    
    # Convert standalone "1" and "2" when they appear to be option numbers
    # More specific patterns to avoid changing unrelated numbers
    content = re.sub(r'\b1\b(?=\s*[:\-\)]|\s*$|\s+[A-ZÆØÅ])', 'A', content)
    content = re.sub(r'\b2\b(?=\s*[:\-\)]|\s*$|\s+[A-ZÆØÅ])', 'B', content)
    
    # Convert common instruction patterns - more comprehensive
    content = re.sub(r'Answer with only the number \(1 or 2\)', 'Answer with only the letter (A or B)', content, flags=re.IGNORECASE)
    content = re.sub(r'Answer with only the number \(1 and 2\)', 'Answer with only the letter (A and B)', content, flags=re.IGNORECASE)
    content = re.sub(r'Answer with only the number 1 or 2', 'Answer with only the letter A or B', content, flags=re.IGNORECASE)
    content = re.sub(r'Answer with only number \(1 or 2\)', 'Answer with only letter (A or B)', content, flags=re.IGNORECASE)
    content = re.sub(r'Select 1 or 2', 'Select A or B', content, flags=re.IGNORECASE)
    content = re.sub(r'Choose 1 or 2', 'Choose A or B', content, flags=re.IGNORECASE)
    content = re.sub(r'Answer 1 or B', 'Answer A or B', content, flags=re.IGNORECASE)
    content = re.sub(r'Answer A or 2', 'Answer A or B', content, flags=re.IGNORECASE)
    content = re.sub(r'Answer 1 or 2', 'Answer A or B', content, flags=re.IGNORECASE)
    content = re.sub(r'Pick 1 or 2', 'Pick A or B', content, flags=re.IGNORECASE)
    content = re.sub(r'number \(1 or 2\)', 'letter (A or B)', content, flags=re.IGNORECASE)
    content = re.sub(r'number 1 or 2', 'letter A or B', content, flags=re.IGNORECASE)
    content = re.sub(r'the number \(1 or B\)', 'the letter (A or B)', content, flags=re.IGNORECASE)
    
    # Handle specific patterns seen in your data
    content = re.sub(r'Your answer \(letter only\):', 'Your answer (letter only):', content, flags=re.IGNORECASE)
    content = re.sub(r'Respond with only A or B:', 'Respond with only A or B:', content, flags=re.IGNORECASE)
    
    # Convert parenthetical references
    content = re.sub(r'\(1\)', '(A)', content)
    content = re.sub(r'\(2\)', '(B)', content)
    
    # Convert patterns like "Answer: (A or B only)"
    content = re.sub(r'Answer:\s*\([AB12]+\s+or\s+[AB12]+\s+only\)', 'Answer: (A or B only)', content, flags=re.IGNORECASE)
    
    # Convert patterns with mixed 1/2 and A/B - ensure consistency
    content = re.sub(r'\(1 or B\)', '(A or B)', content, flags=re.IGNORECASE)
    content = re.sub(r'\(A or 2\)', '(A or B)', content, flags=re.IGNORECASE)
    content = re.sub(r'\(1 or 2\)', '(A or B)', content)
    
    return content

def convert_dpo_to_sft(input_file, output_file, force_ab=True):
    """
    Convert DPO format JSONL file to SFT format.
    Takes the 'chosen' field and renames it to 'messages'.
    Optionally converts 1/2 responses to A/B format in both user and assistant messages.
    """
    processed_lines = 0
    error_count = 0
    assistant_conversions = 0
    user_conversions = 0
    
    print("=" * 70)
    print("CONVERTING DPO FORMAT TO SFT FORMAT")
    print("=" * 70)
    print(f"Task: Extract 'chosen' field and rename to 'messages'")
    print(f"A/B conversion: {'enabled' if force_ab else 'disabled'}")
    print("=" * 70)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📖 Reading data from: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        # Parse the DPO format
                        dpo_data = json.loads(line)
                        
                        # Check if 'chosen' field exists
                        if 'chosen' not in dpo_data:
                            print(f"⚠️  Line {line_num}: Missing 'chosen' field")
                            error_count += 1
                            continue
                        
                        # Get the chosen conversation
                        messages = dpo_data["chosen"]
                        
                        # Validate messages structure
                        if not isinstance(messages, list):
                            print(f"⚠️  Line {line_num}: 'chosen' field is not a list")
                            error_count += 1
                            continue
                        
                        # Convert 1/2 to A/B in messages if force_ab is True
                        if force_ab:
                            for message in messages:
                                if isinstance(message, dict) and message.get("content"):
                                    original_content = message.get("content", "")
                                    
                                    if message.get("role") == "assistant":
                                        # Convert assistant responses
                                        new_content = ensure_ab_format(original_content)
                                        if new_content != original_content:
                                            message["content"] = new_content
                                            assistant_conversions += 1
                                    
                                    elif message.get("role") == "user":
                                        # Convert user content
                                        new_content = convert_user_content_to_ab_format(original_content)
                                        if new_content != original_content:
                                            message["content"] = new_content
                                            user_conversions += 1
                        
                        # Convert to SFT format
                        sft_data = {
                            "messages": messages
                        }
                        
                        # Write to output file
                        json.dump(sft_data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        processed_lines += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON error on line {line_num}: {e}")
                        error_count += 1
                    except Exception as e:
                        print(f"❌ Error processing line {line_num}: {e}")
                        error_count += 1
                
                if line_num % 1000 == 0:
                    print(f"   Processed {line_num} lines...")
    
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        sys.exit(1)
    
    return processed_lines, error_count, assistant_conversions, user_conversions

# Main execution
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"🚀 Process started at: {start_time}")
    
    input_file = os.environ.get('INPUT_FILE')
    output_file = os.environ.get('OUTPUT_FILE')
    no_ab_conversion = os.environ.get('NO_AB_CONVERSION')
    
    # Determine if A/B conversion should be applied
    force_ab = no_ab_conversion != "--no-ab-conversion"
    
    try:
        processed, errors, assistant_conversions, user_conversions = convert_dpo_to_sft(input_file, output_file, force_ab)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n📊 CONVERSION RESULTS:")
        print(f"✅ Successfully processed: {processed} lines")
        if assistant_conversions > 0:
            print(f"🔄 Converted {assistant_conversions} assistant responses from 1/2 format to A/B format")
        if user_conversions > 0:
            print(f"🔄 Converted {user_conversions} user messages from 1/2 format to A/B format")
        if errors > 0:
            print(f"⚠️  Encountered errors: {errors} lines")
        
        print(f"\n⏰ Completed at: {end_time}")
        print(f"⏱️  Duration: {duration}")
        
        # Verify output file
        try:
            output_lines = sum(1 for _ in open(output_file, 'r'))
            print(f"📁 Output file: {output_file}")
            print(f"📄 Output lines: {output_lines}")
            
            # Show example of converted format
            print(f"\n📝 Example SFT format:")
            with open(output_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    example = json.loads(first_line)
                    print(json.dumps(example, ensure_ascii=False, indent=2))
            
        except Exception as e:
            print(f"⚠️  Could not verify output file: {e}")
        
        if errors == 0:
            print(f"\n🎉 SUCCESS! Conversion completed without errors.")
        else:
            print(f"\n⚠️  Conversion completed with {errors} errors.")
        
        print(f"\n📋 Next steps:")
        print(f"   1. Use {output_file} for SFT training")
        print(f"   2. The file is now in messages format compatible with SFT frameworks")
        print(f"   3. Both user questions and assistant responses have been converted to A/B format")
        
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