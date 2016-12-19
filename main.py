"""Main function to run activetm"""
import argparse
import datetime
from email.mime.text import MIMEText
import logging
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from activetm import plot
from activetm import utils

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
    """A class that spins up a thread to launch submain"""
    #pylint:disable-msg=too-many-arguments
    def __init__(self, host, working_dir, settings, outputdir, label):
        threading.Thread.__init__(self)
        self.daemon = True
        self.host = host
        self.working_dir = working_dir
        self.settings = settings
        self.outputdir = outputdir
        self.label = label
        self.killed = False

    # TODO use asyncio when code gets upgraded to Python 3
    def run(self):
        proc = subprocess.Popen(
            [
                'ssh',
                self.host,
                'python3 ' + os.path.join(self.working_dir, 'submain.py') +\
                    ' ' +\
                    self.working_dir + ' ' +\
                    self.settings + ' ' +\
                    self.outputdir + ' ' +\
                    self.label + '; exit 0'])
        while True:
            time.sleep(1)
            if self.killed:
                proc.kill()
                break
            if proc.poll() is not None:
                break


class PickleThread(threading.Thread):
    """A class that spins up a thread to pickle a corpus"""
    #pylint:disable-msg=too-many-arguments
    def __init__(self, host, working_dir, work, outputdir, lock):
        threading.Thread.__init__(self)
        self.daemon = True
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
            subprocess.check_call(
                [
                    'ssh',
                    '-t',
                    self.host,
                    'python3 ' +\
                    os.path.join(self.working_dir, 'pickle_data.py') + ' '+\
                    settings + ' ' +\
                    self.outputdir + '; exit 0'])


def generate_settings(filename):
    """Grab settings files"""
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                yield line


def get_hosts(filename):
    """Get hosts from file"""
    hosts = []
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                hosts.append(line)
    return hosts


def check_counts(hosts, settingscount):
    """Check to see if the right number of hosts have been specified"""
    if len(hosts) != settingscount:
        logging.getLogger(__name__).error('Node count and settings count do not match!')
        sys.exit(1)


def get_groups(config):
    """Get groups (as specified in settings file)"""
    result = set()
    settings = generate_settings(config)
    for setting in settings:
        cur_settings = utils.parse_settings(setting)
        result.add(cur_settings['group'])
    return sorted(list(result))


def pickle_data(hosts, sspaths, working_dir, outputdir):
    """Pickle all corpora needed"""
    picklings = set()
    work = set()
    for setting in generate_settings(sspaths):
        pickle_name = utils.get_pickle_name(setting)
        if pickle_name not in picklings:
            picklings.add(pickle_name)
            work.add(setting)
    lock = threading.Lock()
    threads = []
    for host in set(hosts):
        thread = PickleThread(host, working_dir, work, outputdir, lock)
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def get_corpora(sspaths):
    """Find corpus names"""
    result = set()
    for setting in generate_settings(sspaths):
        corpus = os.path.splitext(
            os.path.basename(
                utils.get_pickle_name(setting)))[0]
        if corpus not in result:
            result.add(corpus)
    return result


def run_jobs(hosts, sspaths, working_dir, outputdir):
    """Run all submain processes

    Also make sure that when main is killed, the subprocesses get killed, too
    """
    threads = []
    try:
        for i, (host, setting) in enumerate(zip(hosts, generate_settings(sspaths))):
            thread = JobThread(host, working_dir, setting, outputdir, str(i))
            thread.daemon = True
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning('Killing children')
        for thread in threads:
            thread.killed = True
        for thread in threads:
            thread.join()
        runningdir = os.path.join(outputdir, 'running')
        for runmark in os.listdir(runningdir):
            parts = runmark.split('.')
            subprocess.call(
                [
                    'ssh',
                    parts[0],
                    'kill -s 9 ' + parts[-1] + '; exit 0'])
        sys.exit(-1)


def extract_data(fpath):
    """Grab results data from file at fpath"""
    data = []
    with open(fpath) as ifh:
        for line in ifh:
            line = line.strip()
            if line and not line.startswith('#'):
                data.append([float(result) for result in line.split()])
    return data


def get_data(dirname):
    """Construct a collection of results data"""
    data = []
    for filename in os.listdir(dirname):
        fpath = os.path.join(dirname, filename)
        if os.path.isfile(fpath):
            data.append(extract_data(fpath))
    return data


#pylint:disable-msg=too-many-locals
def make_plots(outputdir, dirs, deltas):
    """Make plots"""
    # colors = plot.get_separate_colors(len(dirs))
    count_plot = plot.Plotter()
    select_and_train_plot = plot.Plotter()
    time_plot = plot.Plotter()
    mae_plot = plot.Plotter()
    corpus = os.path.basename(outputdir)
    print(corpus)
    dirs.sort()
    for curdirname in dirs:
        curdir = os.path.join(outputdir, curdirname)
        print('\t', curdirname)
        data = np.array(get_data(curdir))
        # for the first document, read off first dimension (the labeled set
        # counts)
        counts = data[0, :, 0]
        # set up a 2D matrix with each experiment's pR^2 results in rows
        ys_mat = data[:, :, -1]
        count_plot.plot(
            counts,
            ys_mat,
            curdirname)
        # set up a 2D matrix with each experiment's time results in rows
        times_mat = data[:, :, 1]
        time_plot.plot(
            times_mat,
            ys_mat,
            curdirname)
        select_and_train_mat = data[:, :, 2]
        select_and_train_plot.plot(
            counts,
            select_and_train_mat,
            curdirname)
        # get mae results
        loss_delta = float(deltas[corpus])
        losses = []
        for maedir in os.listdir(curdir):
            curmaedir = os.path.join(curdir, maedir)
            if os.path.isdir(curmaedir):
                losses.append([])
                for i in range(len(counts)):
                    maedata = np.loadtxt(os.path.join(curmaedir, str(i)))
                    # generalized zero-one loss
                    losses[-1].append(np.sum(maedata < loss_delta) / len(maedata))
        losses = np.array(losses)
        mae_plot.plot(
            counts,
            losses,
            curdirname)
    count_plot.set_xlabel('Number of Labeled Documents')
    count_plot.set_ylabel('pR$^2$')
    count_plot.savefig(os.path.join(outputdir, corpus+'.counts.pdf'))
    time_plot.set_xlabel('Time elapsed (seconds)')
    time_plot.set_ylabel('pR$^2$')
    time_plot.savefig(os.path.join(
        outputdir,
        corpus+'.times.pdf'))
    select_and_train_plot.set_xlabel('Number of Labeled Documents')
    select_and_train_plot.set_ylabel('Time to select and train')
    select_and_train_plot.savefig(os.path.join(
        outputdir,
        corpus+'.select_and_train.pdf'))
    mae_plot.set_xlabel('Number of Labeled Documents')
    mae_plot.set_ylabel('Zero-One Loss, '+str(loss_delta))
    mae_plot.savefig(os.path.join(outputdir, corpus+'.zero_one_loss.pdf'))
    for curplot in [count_plot, time_plot, select_and_train_plot, mae_plot]:
        curplot.close()


def send_notification(email, outdir, run_time):
    """Send e-mail notification"""
    msg = MIMEText('Run time: '+str(run_time))
    msg['Subject'] = 'Experiment Finished for '+outdir
    msg['From'] = email
    msg['To'] = email

    proc = os.popen('/usr/sbin/sendmail -t -i', 'w')
    proc.write(msg.as_string())
    status = proc.close()
    if status:
        logging.getLogger(__name__).warning('sendmail exit status '+str(status))


def slack_notification(msg):
    """Send slack notification"""
    slackhook = 'https://hooks.slack.com/services/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'
    payload = 'payload={"channel": "#potatojobs", "username": "potatobot", ' +\
            '"text": "'+msg+'", "icon_emoji": ":fries:"}'
    subprocess.call([
        'curl',
        '-X',
        'POST',
        '--data-urlencode',
        payload,
        slackhook])


def _parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(
        description='Launcher for ActiveTM experiments')
    parser.add_argument(
        'hosts',
        help='hosts file for job farming')
    parser.add_argument(
        'working_dir',
        help='ActiveTM directory available to hosts (should be a network path)')
    parser.add_argument(
        'config',
        help=\
            '''a file with the path to a settings file on each line.
            The file referred to should follow the settings specification
            as discussed in README.md in the root ActiveTM directory''')
    parser.add_argument(
        'deltastxt',
        help='file of deltas for computing mean absolute error')
    parser.add_argument(
        'outputdir',
        help='directory for output (should be a network path)')
    parser.add_argument(
        'email',
        help='email address to send to when job completes',
        nargs='?')
    return parser.parse_args()


def _run():
    """Run code"""
    args = _parse_args()
    try:
        begin_time = datetime.datetime.now()
        slack_notification('Starting job: '+args.outputdir)
        runningdir = os.path.join(args.outputdir, 'running')
        if os.path.exists(runningdir):
            shutil.rmtree(runningdir)
        try:
            os.makedirs(runningdir)
        except OSError:
            pass
        hosts = get_hosts(args.hosts)
        check_counts(hosts, utils.count_settings(args.config))
        if not os.path.exists(args.outputdir):
            logging.getLogger(__name__).error('Cannot write output to: '+args.outputdir)
            sys.exit(-1)
        groups = get_groups(args.config)
        pickle_data(hosts, args.config, args.working_dir, args.outputdir)
        run_jobs(
            hosts,
            args.config,
            args.working_dir,
            args.outputdir)
        corpora = get_corpora(args.config)
        for corpus in corpora:
            make_plots(
                os.path.join(args.outputdir, corpus),
                groups,
                utils.parse_settings(args.deltastxt))
        run_time = datetime.datetime.now() - begin_time
        with open(os.path.join(args.outputdir, 'run_time'), 'w') as ofh:
            ofh.write(str(run_time))
        os.rmdir(runningdir)
        slack_notification('Job complete: '+args.outputdir)
        if args.email:
            send_notification(args.email, args.outputdir, run_time)
    except:
        slack_notification('Job died: '+args.outputdir)
        raise

if __name__ == '__main__':
    _run()

