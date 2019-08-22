#!/usr/bin/env python3

import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Plot grade distribution of Moodle quizzes.')
parser.add_argument('-g', '--grades_file', required=True,
                    help='Input file with quiz grades from Moodle.')
parser.add_argument('-o', '--output', required=True,
                    help='Output file (PDF) with distribution plots.')
args = parser.parse_args()

# Read grades file
df = pd.read_csv(args.grades_file, sep=',')

# Compute max points per columns and in total
max_points = {}
columns = {}

for c in df.columns:
    match = re.match('^Q\. (\d+) /(\d+\.\d\d)$', c)
    if match is not None:
        q = int(match.group(1))
        max_points[q] = float(match.group(2))
        columns[q] = c

for c in columns.values():
    df[c] = df[c].replace(['-'], 0.0)
    df[c] = df[c].astype(float)

grade_column = [c for c in df.columns if c.startswith('Grade/')]
assert len(grade_column) == 1
grade_column = grade_column.pop()
total_points = float(re.match('^Grade/(\d+\.\d\d)$', grade_column).group(1))

# Produce plots
with PdfPages(args.output) as pdf:
    # Total
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with sns.plotting_context("notebook", font_scale=1):
        sns.distplot(df[grade_column], label='Total',
                     kde=True, hist=True, ax=ax, bins=20)
        ax.axvline(total_points)
    pdf.savefig(bbox_inches='tight')

    # Per question
    for (i, c) in columns.items():
        p = max_points[i]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if len(df[c].unique()) <=1: continue
        sns.distplot(df[c],
                     axlabel='Question {0} ({1} points)'.format(i, p),
                     kde=True, hist=True, ax=ax)
        ax.axvline(p)
        pdf.savefig(bbox_inches='tight')
