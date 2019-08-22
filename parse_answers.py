#!/usr/bin/env python3

import argparse
from collections import Counter
import io
from os.path import commonprefix
import pandas as pd
import pytrie
import re
import subprocess

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Parse answers of Moodle quizzes.')
parser.add_argument('-r', '--responses_file', type=argparse.FileType('r'),
                    help='Input file with quiz responses from Moodle.',
                    required=True)
parser.add_argument('-g', '--grades_file', type=argparse.FileType('r'),
                    help='Input file with quiz grades from Moodle.',
                    required=True)
parser.add_argument('-e', '--excel',
                    help='Output file (Excel format) with parsed quiz answers.')
parser.add_argument('-c', '--csv',
                    help='Output file (CSV format) with parsed quiz answers.')
parser.add_argument('-S', '--Stats',
                    help='Output file (Excel format) with statistics.')
parser.add_argument('-s', '--stats',
                    help='Output file (CSV format) with statistics.')
parser.add_argument('-q', '--check-question', nargs='+',
                    help='Label of question(s) that should be checked.')
parser.add_argument('-x', '--external-check', nargs='+',
                    help='External command that is run to check the response.')
args = parser.parse_args()

# Fix special characters
responses_file = io.StringIO(args.responses_file.read()
        .replace('\u00A0', ' ')
        .replace('\xc2', ' '))
grades_file = io.StringIO(args.grades_file.read()
        .replace('\u00A0', ' ')
        .replace('\xc2', ' '))

# Read data from CSV
df_responses = pd.read_csv(responses_file)
df_grades = pd.read_csv(grades_file)
df_parts = pd.DataFrame()
df_right_answers = pd.DataFrame()

# Helper to get the number of parts of a Cloze question
def compute_num_parts(s):
    s = str(s)
    for term in ['part', 'Teil']:
        parts = [int(m.group(1)) for m in re.finditer(term + ' ([0-9]+):', s)]
        num_parts = max([0] + parts)
        if num_parts > 0:
            return num_parts
    return 0

# Helper to get part i of a Cloze question with n parts
def extract_part(s, i, n):
    s = str(s)
    for term in ['part', 'Teil']:
        pattern1 = '{0} {1}: +(.*)'
        pattern2 = '(; {0} {2}: )' if i < n else ''
        pattern = (pattern1 + pattern2).format(term, i, i + 1)
        m = re.search(pattern, s)
        if m:
            return m.group(1)

# Iterate through columns and try to parse each of them
for c in [c for c in df_responses.columns if c.startswith('Response ')]:
    print("Attempting to parse as Cloze question", c);
    num_parts = sorted(set(df_responses[c].apply(compute_num_parts)))
    if len(num_parts) == 1 and num_parts[0] != 0:
        num_parts = num_parts[0]

        print("  Success! Detected {0} parts. Parsing...".format(num_parts))
 
        for i in range(1, num_parts + 1):
            new_response = '{0}: part {1}'.format(c, i)
            df_parts[new_response] = df_responses[c] \
                    .apply(lambda x: extract_part(x, i, num_parts))

        for i in range(1, num_parts + 1):
            right_answer_column = 'Right answer' + c[len('Response'):]
            right_answer_part = '{0}: part {1}'.format(right_answer_column, i)
            df_right_answers[right_answer_part] = \
                df_responses[right_answer_column] \
                    .apply(lambda x: extract_part(x, i, num_parts))
        continue
    print("  Not a Cloze question. Parts found: ", num_parts)

def shorten_labels(labels):
    # Base cases
    if len(labels) == 0:
        return labels
    if max(map(len, labels)) <= 3:
        return labels
    if len(labels) == 1:
        if len(labels[0]) > 12:
            return [labels[0][0:9] + '...']
        return [labels[0]]

    # Remove common prefix and shorten if necessary
    common = commonprefix(labels)
    suffixes = [l[len(common):] for l in labels]
    if len(common) > 5:
        common = common[0:5] + '...' + common[-5:]
    suffixes = pytrie.StringTrie.fromkeys(suffixes)

    # Partition by next 3 characters and recurse
    shortened_suffixes = set()
    next_radixes = set((l[0:3] for l in suffixes))
    for r in next_radixes:
        shortened_suffixes.update(shorten_labels(suffixes.keys(prefix=r)))

    # Concat shortened labels with removed prefix
    return [common + s for s in shortened_suffixes]

def extract_selections(answer):
    l = re.findall('\{[^\}]*\}', answer)
    if ' '.join(l) == answer:
        return list(l)
    return None

def compute_num_selections(answer):
    selections = extract_selections(answer) or []
    return len(selections)

def extract_selection(answer, i):
    selections = extract_selections(answer)
    if selections is None:
        return None
    return selections[i]

for c in [c for c in df_responses.columns if c.startswith('Right answer ')]:
    print("Attempting to parse as 'select missing words' question", c)

    column_i = c[len('Right answer '):]
    column_answer = 'Right answer ' + column_i
    column_response = 'Response ' + column_i

    df_responses[column_answer].fillna('', inplace=True)
    df_responses[column_response].fillna('', inplace=True)

    num_selections = df_responses[column_answer] \
        .apply(lambda x: compute_num_selections(x)) \
        .unique().tolist()

    if len(num_selections) != 1 or num_selections[0] == 0:
        print("  No selections found.")
        continue

    num_selections = num_selections[0]
    print("  Success! {} gaps found.".format(num_selections))

    for i in range(num_selections):
        column_part = '{0}: part {1}'.format(column_response, i + 1)
        if column_part in df_parts.columns:
            print("  Already parsed previously.")
            break
        df_parts[column_part] = \
                df_responses[column_response] \
                        .apply(lambda r: extract_selection(r, i))

        answer_column_part = '{0}: part {1}'.format(column_answer, i + 1)
        df_right_answers[answer_column_part] = \
                df_responses[column_answer] \
                        .apply(lambda r: extract_selection(r, i))

def compute_options(s):
    if not type(s) is str:
        return None
    parts = s.split('; ')
    if any(len(p.split(': ')) != 2 for p in parts):
        return None
    result = []
    for part in parts:
        part = part.split(': ')[0]
        if part.startswith('Teil '):
            part = 'part ' + part[len('Teil '):]
        result.append(part)
    return sorted(result)

def extract_choice(row, i):
    options = compute_options(row.iloc[0])
    if not row.iloc[1]:
        return None
    pattern = '{0}: ([^;]*)(;.*)?'.format(re.escape(options[i]))
    m = re.search(pattern, row.iloc[1])
    if m:
        return m.group(1)

for c in [c for c in df_responses.columns if c.startswith('Right answer ')]:
    print("Attempting to parse as generic question with parts", c)

    column_i = c[len('Right answer '):]
    column_answer = 'Right answer ' + column_i
    column_response = 'Response ' + column_i

    df_responses[column_answer].fillna('', inplace=True)
    df_responses[column_response].fillna('', inplace=True)

    options = compute_options(df_responses[column_answer][0])
    if not options:
        print("  No parts found.")
        continue
    options = shorten_labels(options)
    print("  Found parts: ", options)

    for i, o in enumerate(options):
        column_part = '{0}: {1}'.format(column_response, o)
        if column_part in df_parts.columns:
            print("  Already parsed previously.")
            break
        df_parts[column_part] = \
                df_responses[[column_answer, column_response]] \
                        .apply(lambda r: extract_choice(r, i), axis=1)

        answer_column_part = '{0}: {1}'.format(column_answer, o)
        df_right_answers[answer_column_part] = \
                df_responses[[column_answer, column_answer]] \
                        .apply(lambda r: extract_choice(r, i), axis=1)

def extract_mappings(answer):
    answers = answer.split('; ')
    return dict([a.split(' -> ') for a in answers])

def extract_mapping(answer, i):
    try:
        answers = extract_mappings(answer)
    except BaseException:
        return None
    return list(answers.values())[i]

for c in [c for c in df_responses.columns if c.startswith('Right answer ')]:
    print("Attempting to parse as mapping question", c)

    parsed_questions = set((c.split(':')[0] for c in df_right_answers.columns))
    if c in parsed_questions:
        print("  Already parsed previously.")
        continue

    column_i = c[len('Right answer '):]
    column_answer = 'Right answer ' + column_i
    column_response = 'Response ' + column_i

    df_responses[column_answer].fillna('', inplace=True)
    df_responses[column_response].fillna('', inplace=True)

    mappings = None
    try:
        mappings = df_responses[column_answer] \
            .apply(lambda a: str(extract_mappings(a))) \
            .unique().tolist()
    except BaseException:
        pass
    if mappings is None or len(mappings) != 1:
        print("  No mapping found.")
        continue

    mappings = extract_mappings(df_responses[column_answer].iloc[0])
    mappings = dict(zip(shorten_labels(mappings.keys()), mappings.values()))

    print(" Success! Found mappings:", list(mappings.keys()))

    for i, m in enumerate(mappings.keys()):
        column_part = '{0}: {1}'.format(column_response, m)
        df_parts[column_part] = \
                df_responses[column_response] \
                        .apply(lambda r: extract_mapping(r, i))

        answer_column_part = '{0}: {1}'.format(column_answer, m)
        df_right_answers[answer_column_part] = \
                df_responses[column_answer] \
                        .apply(lambda r: extract_mapping(r, i))

for c in [c for c in df_responses.columns if c.startswith('Right answer ')]:
    print("Attempting to parse as generic question", c)

    parsed_questions = set((c.split(':')[0] for c in df_right_answers.columns))
    if c in parsed_questions:
        print("  Already parsed previously.")
        continue

    column_i = c[len('Right answer '):]
    column_answer = 'Right answer ' + column_i
    column_response = 'Response ' + column_i

    df_responses[column_answer].fillna('', inplace=True)
    df_responses[column_response].fillna('', inplace=True)

    responses = df_responses[column_response] \
            .replace(['Wahr', 'Falsch'], ['True', 'False'])

    print("  Listing all {} given answers...".format(
        len(responses.unique().tolist())))

    df_parts[column_response + ': (unique part)'] = responses
    df_right_answers[column_answer + ': (unique part)'] = \
            df_responses[column_answer] \
                    .replace(['Wahr', 'Falsch'], ['True', 'False'])

# Run external checks
if bool(args.check_question) ^ bool(args.external_check):
    raise ValueError('Only one of check label and check command given')
if args.check_question:
    if len(args.check_question) != len(args.external_check):
        raise ValueError('Different numbers of labels and check commands given')

    for (question, command) in zip(args.check_question, args.external_check):
        df_responses[question + " result"] = df_responses[question] \
                .apply(lambda x: "" if x is None else x) \
                .apply(lambda x: subprocess.call(command.format(x)))

# Produce stats
df_stats = pd.DataFrame()
for c in sorted(df_parts.columns,
                key=lambda s: int(re.search('Response (\d+):', s).group(1))):
    right_answer_column = 'Right answer' + c[len('Response'):]
    right_answer = df_right_answers[right_answer_column]
    assert len(set(right_answer)) == 1
    right_answer = set(right_answer).pop()

    frequencies = dict(Counter(df_parts[c]))
    answers = [(right_answer, frequencies.pop(right_answer, 0))]
    answers += sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

    answers = [(a, n, int(n/len(df_parts.index) * 100 * 10) / 10)
               for (a, n) in answers]
    columns = ['{} ({})'.format(c, metric)
               for metric in ['answer', 'number', 'frequency']]
    df_stats = pd.concat([df_stats, pd.DataFrame(answers, columns=columns)],
                         axis=1)
if args.excel:
    df_stats.to_excel(args.Stats)
elif args.csv:
    df_stats.to_csv(args.stats)

# Join responses and grades
df_responses = pd.concat([df_responses, df_parts], axis=1)
columns_to_drop = [
    'Surname', 'First name', 'Matriculation number', 'Institution',
    'Department', 'State', 'Started on', 'Completed', 'Time taken',
]
columns_to_drop = [c for c in columns_to_drop if c in df_grades.columns]
df_grades.drop(axis=1, inplace=True, labels=columns_to_drop)
df = df_responses.merge(df_grades, on='Email address')

# Save result
if args.excel:
    df.to_excel(args.excel)
elif args.csv:
    df.to_csv(args.csv)
else:
    raise ValueError('Need to specify CSV or Excel output file')
