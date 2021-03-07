## profiling with cProfile

python -m cProfile -o profile.txt run_kmeans.py
python analyze_stats.py > analysis_after.txt
