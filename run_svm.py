import argparse
import subprocess
import sys
import os

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {command}", file=sys.stderr)
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--users', type=int, default=100)
    parser.add_argument('--rows', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--attackers', type=int, default=2000)
    parser.add_argument('--data', type=str, default='../GAN/output_with_trust_scores.csv',
                        help='Path to the input data file for SVM')
    args = parser.parse_args()

    python_cmd = 'python.exe' if os.name == 'nt' else 'python'

    run_command(f"cd GenerateData && {python_cmd} generate.py {args.users} {args.rows}")
    run_command(f"cd GAN && {python_cmd} gan.py {args.samples}")
    run_command(f"cd SVM && {python_cmd} split.py {args.attackers} {args.data}")
    run_command(f"cd SVM && {python_cmd} svm.py")
    run_command(f"cd SVM && {python_cmd} fpr_fnr.py")

if __name__ == '__main__':
    main()
