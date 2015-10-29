from __future__ import division
import argparse
import numpy as np
import datetime
from email.mime.text import MIMEText
import os
import subprocess
import sys
import threading

import ankura

import activetm.labeled
import activetm.plot as plot
import activetm.utils as utils

'''
The output from an experiment should take the following form:

    output_directory
        settings1
            run1_1
            run2_1
            ...
        settings2
        ...

In this way, it gets easier to plot the results, since each settings will make a
line on the plot, and each line will be aggregate data from multiple runs of the
same settings.
'''

class JobThread(threading.Thread):
    def __init__(self, host, working_dir, settings, outputdir, label):
        threading.Thread.__init__(self)
        self.host = host
        self.working_dir = working_dir
        self.settings = settings
        self.outputdir = outputdir
        self.label = label

    def run(self):
        subprocess.check_call(['ssh',
            self.host,
            'python ' + os.path.join(self.working_dir, 'submain.py') + ' ' +\
                            self.working_dir + ' ' +\
                            self.settings + ' ' +\
                            self.outputdir + ' ' +\
                            self.label + '; exit 0'])

class PickleThread(threading.Thread):
    def __init__(self, host, working_dir, work, outputdir, lock):
        threading.Thread.__init__(self)
        self.host = host
        self.working_dir = working_dir
        self.work = work
        self.outputdir = outputdir
        self.lock = lock

    def run(self):
        while True:
            with self.lock:
                if len(self.work) <= 0:
                    break
                else:
                    settings = self.work.pop()
            print 'Pickling: ', settings
            subprocess.check_call(['ssh',
                self.host,
                'python ' + os.path.join(self.working_dir, 'pickle_data.py') + ' '+\
                        settings + ' ' +\
                        self.outputdir + '; exit 0'])

def generate_settings(filename):
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                yield line

def get_hosts(filename):
    hosts = []
    with open(args.hosts) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                hosts.append(line)
    return hosts

def check_counts(hosts, settingscount):
    if len(hosts) != settingscount:
        print 'Node count and settings count do not match!'
        sys.exit(1)

def get_groups(config):
    result = set()
    settings = generate_settings(config)
    for s in settings:
        d = utils.parse_settings(s)
        result.add(d['group'])
    return sorted(list(result))

def pickle_data(hosts, settings, working_dir, outputdir):
    work = set()
    for s in settings:
        if not os.path.exists(utils.get_pickle_name(s)):
            work.add(s)
    lock = threading.Lock()
    threads = []
    for h in set(hosts):
        t = PickleThread(h, working_dir, work, outputdir, lock)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def run_jobs(hosts, settings, working_dir, outputdir):
    threads = []
    for h, s, i in zip(hosts, settings, range(len(hosts))):
        t = JobThread(h, working_dir, s, outputdir, str(i))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def extract_data(fpath):
    data = []
    with open(fpath) as ifh:
        for line in ifh:
            line = line.strip()
            if line and not line.startswith('#'):
                results = line.split()
                if len(data) == 0:
                    for _ in range(len(results)):
                        data.append([])
                for i, r in enumerate(results):
                    data[i].append(float(r))
    return data

def get_data(dirname):
    data = []
    for f in os.listdir(dirname):
        fpath = os.path.join(dirname, f)
        if os.path.isfile(fpath):
            data.append(extract_data(fpath))
    return data

def get_stats(mat):
    # compute the medians along the columns
    mat_medians = np.median(mat, axis=0)
    # compute the means along the columns
    mat_means = np.mean(mat, axis=0)
    # find difference of means from first quartile along the columns
    mat_errs_minus = mat_means - np.percentile(mat, 25, axis=0)
    # compute third quartile along the columns; find difference from means
    mat_errs_plus = np.percentile(mat, 75, axis=0) - mat_means
    return mat_medians, mat_means, mat_errs_plus, mat_errs_minus

def make_plots(outputdir, dirs):
    colors = plot.get_separate_colors(len(dirs))
    dirs.sort()
    count_plot = plot.Plotter(colors)
    time_plot = plot.Plotter(colors)
    for d in dirs:
        data = np.array(get_data(os.path.join(outputdir, d)))
        # for the first document, read off first dimension (the labeled set
        # counts)
        counts = data[0,0,:]
        # set up a 2D matrix with each experiment on its own row and each
        # experiment's pR^2 results in columns
        ys_mat = data[:,2,:]
        ys_medians, ys_means, ys_errs_minus, ys_errs_plus = get_stats(ys_mat)
        # set up a 2D matrix with each experiment on its own row and each
        # experiment's time results in columns
        times_mat = data[:,1,:]
        times_medians, times_means, times_errs_minus, times_errs_plus = \
                get_stats(times_mat)
        count_plot.plot(counts, ys_means, d, ys_medians, [ys_errs_minus,
            ys_errs_plus])
        time_plot.plot(times_means, ys_means, d, ys_medians, [ys_errs_minus,
            ys_errs_plus], times_medians, [times_errs_minus, times_errs_plus])
    count_plot.set_xlabel('Number of Labeled Documents')
    count_plot.set_ylabel('pR$^2$')
    count_plot.savefig('counts.pdf')
    time_plot.set_xlabel('Time elapsed (seconds)')
    time_plot.set_ylabel('pR$^2$')
    time_plot.savefig('times.pdf')

def send_notification(email, outdir, run_time):
    msg = MIMEText('Run time: '+str(run_time))
    msg['Subject'] = 'Experiment Finished for '+outdir
    msg['From'] = email
    msg['To'] = email

    p = os.popen('/usr/sbin/sendmail -t -i', 'w')
    p.write(msg.as_string())
    status = p.close()
    if status:
        print "sendmail exit status", status

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher for ActiveTM '
            'experiments')
    parser.add_argument('hosts', help='hosts file for job '
            'farming')
    parser.add_argument('working_dir', help='ActiveTM directory '
            'available to hosts (should be a network path)')
    parser.add_argument('config', help=\
            '''a file with the path to a settings file on each line.
            The file referred to should follow the settings specification
            as discussed in README.md in the root ActiveTM directory''')
    parser.add_argument('outputdir', help='directory for output (should be a '
            'network path)')
    parser.add_argument('email', help='email address to send to when job '
            'completes', nargs='?')
    args = parser.parse_args()

    begin_time = datetime.datetime.now()
    hosts = get_hosts(args.hosts)
    check_counts(hosts, utils.count_settings(args.config))
    if not os.path.exists(args.outputdir):
        print 'Cannot write output to: ', args.outputdir
        sys.exit(-1)
    groups = get_groups(args.config)
    pickle_data(hosts, generate_settings(args.config), args.working_dir, args.outputdir)
    run_jobs(hosts, generate_settings(args.config), args.working_dir,
            args.outputdir)
    make_plots(args.outputdir, groups)
    run_time = datetime.datetime.now() - begin_time
    with open(os.path.join(args.outputdir, 'run_time'), 'w') as ofh:
        ofh.write(str(run_time))
    if args.email:
        send_notification(args.email, args.outputdir, run_time)

