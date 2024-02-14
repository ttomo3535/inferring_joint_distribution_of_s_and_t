import numpy as np
import pandas as pd
import csv
import multiprocessing
import pyper
from my_module import forward_trajectory as fwd
from my_module import mstools
from my_module import mbslib
import functools
import os
import itertools
import random


def make_trajectoryfiles_forward(N0, generation, demography_in_year, t_mutation_in_year, s, h, resolution,
                         n_trajectory, path_to_traj):
    '''
    generate trajectory files

    Args:
        N0 (int):
        generation (int): generation time, years/generation
        demography_in_year (list): demographic history/
        t_mutation_in_year (float): time when mutation arises, in year
        s,
        h,
        resolution,
        n_trajectory (int): number of trajectories
        path_to_traj (str) : path to trajectory files (w/o extentions)

    '''
    for i in range(n_trajectory):

        # file name
        filename = '{}_{}.dat'.format(path_to_traj, i)

        # generate trajectory
        trajectory = fwd.mbs_input(t_mutation_in_year,
                                   demography_in_year,
                                   s, h,
                                   generation, N0, resolution,
                                   'NOTLOST')

        # save
        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for freq in trajectory:
                writer.writerow(freq)


def run_mbs(nsam, per_site_theta, per_site_rho,
            lsites, selpos,
            n_trajectory, nrep_per_traj,
            path_to_mbs_output, path_to_traj):
    '''
    run mbs
    Args:
        nsam,
        per_site_theta,
        per_site_rho,
        lsites,
        selpos,
        n_trajectory (int): number of trajectory files
        nrep_per_traj (int): number of simulations per trajectory file
        path_to_mbs_output (str) : path to mbs output files (w/o extentions)
        path_to_traj (str) : path to trajectory files (w/o extentions)
    '''

    cmd = 'mbs {} -t {} -r {} '.format(nsam, per_site_theta, per_site_rho)
    cmd += '-s {} {} '.format(lsites, selpos)
    cmd += '-f {} {} {} '.format(n_trajectory, nrep_per_traj, path_to_traj)
    cmd += '> {}'.format(path_to_mbs_output)

    mbslib.run_command(cmd)


def parameter_sets_forward(data_num):
    params = dict()

    params['N0'] = 5000
    params['generation'] = 20
    params['demography_in_year'] = [[0, 100 * params['N0'] * params['generation'], params['N0']]]

    # selection coefficient (temp value)
    params['s'] = 0
    params['h'] = 0.5  # <--- co-dominance
    params['resolution'] = 100

    # number of trajectory
    params['n_trajectory'] = 1
    # coalescent simulation per trajectory
    params['nrep_per_traj'] = 1

    # number of chromosome
    params['nsam'] = 120
    # length of sequence
    params['lsites'] = 1000000
    # position of target site
    params['selpos'] = params['lsites']/2

    # mutation rate per site per generation
    params['per_site_theta'] = 1.0 * 10 ** (-8) * 4 * params['N0']
    # recombination rate per site per generation
    params['per_site_rho'] = 1.0 * 10 ** (-8) * 4 * params['N0']

    # mutation age in year (temp value)
    params['t_mutation_in_year'] = 10000
    #
    params_list = list()
    for i in data_num:
        params['data_num'] = i
        params_list.append(params.copy())

    return params_list


def extract_region(input_file, output_file, windowsize, cmd, lsites):
    # assuming target site is lsites/2
    # dummy allele
    allele = '//0-1allele: '
    with open(output_file, 'w') as f:
        f.write(cmd + "\n")
        f.write("\n")
        f.write(allele + "\n")
        for h in mbslib.parse_mbs_data(input_file):
            # extract edge pos
            m = [i for i, x in enumerate(h['pos']) if x < (lsites / 2 - windowsize / 2)]
            p = [i for i, x in enumerate(h['pos']) if x > (lsites / 2 + windowsize / 2)]
            # extract position within window
            if len(m) == 0:
                left_edge = 0
            else:
                left_edge = m[-1] + 1
            if len(p) == 0:
                right_edge = None
            else:
                right_edge = p[0]

            #print('left edge is', left_edge)
            #print('right edge is', right_edge)
            pos = [str(int(i)) for i in h['pos'][left_edge:right_edge]]
            #print(pos)

            # extract SNP within window
            int_site_list = [[int(i) for i in list(j)] for j in h['seq']]
            int_site_list = [list(x) for x in zip(*int_site_list)]
            int_site_list = int_site_list[left_edge:right_edge]
            int_site_list = [list(x) for x in zip(*int_site_list)]
            str_site_list = [list(map(str, row)) for row in int_site_list]
            str_site_list = ["".join(i) for i in str_site_list]

            # number of SNPs
            f.write("segsites: {}\n".format(len(pos)))

            # pos data
            f.write("positions: ")
            f.write(" ".join(pos))
            f.write("\n")

            # SNP data
            f.write("\n".join(str_site_list))
            f.write("\n\n")


def mbs2msoutput(mbs_input_file, ms_output_file, nsam, n_traj,
                 per_site_theta, per_site_rho, selpos, lsites):
    # generate file
    with open(ms_output_file, 'w') as f:
        # convert mbs format to ms format
        f.write("ms {} {} -t {} -r {} {}\n\n".format(nsam, n_traj,
                                                     per_site_theta * lsites,
                                                     per_site_rho * lsites, lsites))

        # convert into ms format
        for i in mbslib.parse_mbs_data(mbs_input_file):
            h = mbslib.mbs_to_ms_output(i, selpos, lsites)
            f.write("//\n")
            # write segregating sites
            f.write("segsites: {}\n".format(len(h['pos'])))

            # write position
            f.write("positions: ")
            # convert int to str
            pos_list = [str(i) for i in h['pos']]
            f.write(" ".join(pos_list))
            f.write("\n")

            # write seq
            f.write("\n".join(h["seq"]))
            f.write("\n\n")


def calc_EHH_sfs(params, data_dir, stats_dir, n_run, windowsize, distance):

    stats_list = []
    def log_uniform_sample(low, high, size=None):
        log_low = np.log(low)
        log_high = np.log(high)

        return np.exp(np.random.uniform(log_low, log_high, size=size))

    # set parameter
    s_classes = [1, 2, 3]
    t_classes = [1, 2, 3]
    classes = list(itertools.product(s_classes, t_classes))
    #weights = [1, 1, 1, 1, 2, 1, 1, 1, 1]
    #weights = list(np.outer(a, b).flatten())
    weights = [9, 6, 9, 6, 8, 6, 9, 6, 9]
    
    np.random.seed(os.getpid())
    for i in range(n_run):
        # define s class
        '''
        t_bins = [1000, 12500, 25000, 37500, 50000, 62500, 75000, 87500, 100000]
        s_bins = [0.0001, 0.000316, 0.001, 0.00316, 0.01, 0.0316, 0.1,  0.316, 1]
        '''
        s_class, t_class = random.choices(classes, weights=weights)[0]

        if s_class == 1:
            # define s
            low = 0.0001
            high = 0.00316
            size = 1
            params['s'] = log_uniform_sample(low, high, size=size)[0]

        elif s_class == 2:
            # define s
            low = 0.00316
            high = 0.0316
            size = 1
            params['s'] = log_uniform_sample(low, high, size=size)[0]

        elif s_class == 3:
            # define s
            low = 0.0316
            high = 1
            size = 1
            params['s'] = log_uniform_sample(low, high, size=size)[0]

        if t_class == 1:
            # define s
            low = 1000
            high = 12500*3
            size = 1
            params['t_mutation_in_year'] = random.randint(low, high)

        elif t_class == 2:
            # define s
            low = 12500*3
            high = 12500*5
            size = 1
            params['t_mutation_in_year'] = random.randint(low, high)

        elif t_class == 3:
            # define s
            low = 12500*5
            high = 12500*8
            size = 1
            params['t_mutation_in_year'] = random.randint(low, high)

        #print('t is', params['t_mutation_in_year'], 's is', params['s'])

        # path to trajectory file
        path_to_traj = "{}/traj_t{}_s{}_datanum{}" \
            .format(data_dir, params['t_mutation_in_year'], params['s'], params['data_num'])

        # path to mbs output
        path_to_mbs_output = "{}/mbs_nsam{}_tmutation{}_s{}_datanum{}.dat" \
            .format(data_dir, params['nsam'], params['t_mutation_in_year'], params['s'], params['data_num'])

        # trajectory file
        make_trajectoryfiles_forward(params['N0'], params['generation'],
                                     params['demography_in_year'], params['t_mutation_in_year'],
                                     params['s'], params['h'], params['resolution'],
                                     params['n_trajectory'], path_to_traj)

        # run mbs
        run_mbs(params['nsam'], params['per_site_theta'], params['per_site_rho'],
                params['lsites'], params['selpos'],
                params['n_trajectory'], params['nrep_per_traj'],
                path_to_mbs_output, path_to_traj)

        # extract window
        mbs_sfs_file = "{}/mbs_nsam{}_tmutation{}_s{}_datanum{}_extracted.dat" \
            .format(data_dir, params['nsam'], params['t_mutation_in_year'], params['s'], params['data_num'])

        # dumy cmd
        cmd = 'command: '
        cmd += 'mbs {} -t {} -r {} '.format(params['nsam'], params['per_site_theta'], params['per_site_rho'])
        cmd += '-s {} {} '.format(params['lsites'], params['selpos'])
        cmd += '-f {} {} {} '.format(params['n_trajectory'], params['nrep_per_traj'], path_to_traj)

        extract_region(path_to_mbs_output, mbs_sfs_file, windowsize, cmd, params['lsites'])
        '''
                try:
            # extract window3
            extract_region(path_to_mbs_output, output_file, windowsize, cmd, params['lsites'])
        except Exception as e:
            print("An error occurred:", print('t is', params['t_mutation_in_year'], 's is', params['s']))
        '''
        # calc SFS
        theta_list = [mstools.calc_thetapi_S_thetaw_thetal(m['seq']) for m in
                      mbslib.parse_mbs_data(mbs_sfs_file)]

        if len(theta_list) == 0:
            # stats
            D, H = np.nan, np.nan

        else:
            D = mstools.calc_D(*theta_list[0], params['nsam'])
            H = mstools.calc_H(*theta_list[0], params['nsam'])

        # extract current frequency
        traj_file = f"{path_to_traj}_0.dat"
        dt = pd.read_table(traj_file, header=None)
        d_freq = dt.iloc[0, 3]

        # stats
        sfs_stats = [D, H]
        traj = [params['t_mutation_in_year'], params['s'], d_freq]

        # calc_EHH
        # convert mbs into ms
        ms_ehh_file = '{}/ms_t{}_s{}_datanum{}.txt' \
            .format(data_dir, params['t_mutation_in_year'], params['s'], params['data_num'])

        mbs2msoutput(path_to_mbs_output, ms_ehh_file, params['nsam'], params['n_trajectory'],
                     params['per_site_theta'], params['per_site_rho'], params['selpos'], params['lsites'])

        # extract the position of the target site
        mrk = [pos.index('0.5') + 1 for m in mstools.parse_ms_data(ms_ehh_file) for pos in [m['pos']]][0]
        # print('mrk is', mrk)

        # run R script to calculate EHH
        r = pyper.R(use_numpy='True', use_pandas='True')
        r.assign('ms_file', ms_ehh_file)
        r("mrk <- {}".format(mrk))
        r("distance <- {}".format(distance))
        r("nrep <- {}".format(params['n_trajectory']))
        r("lsites <- {}".format(params['lsites']))
        r("source(file='calc_ehh_pyper.R')")

        # get EHH data
        mrk = r.get('mrk')
        print('mrk is', mrk)
        ehh_stats = r.get('stats_list')
        #print('data type is', type(ehh_stats))
        #print('iHS is', ehh_stats)
        # combine ehh data and sfs data
        if len(ehh_stats)==0:
            ehh_stats = [np.nan]*10
            
        else:
            ehh_stats = ehh_stats[0]

        #print('sfs_stats is ', sfs_stats, ', ehh_stat-')
        stats = sfs_stats + ehh_stats + traj
        #print(stats)
        stats_list.append(stats)
        # delete files
        os.remove(path_to_mbs_output)
        os.remove(traj_file)
        os.remove(mbs_sfs_file)
        os.remove(ms_ehh_file)
        #print('data num is', params['data_num'], 't is', traj[0], 's is', traj[1])
    # save
    columns = ['D', 'H', 'EHH_A_minus', 'EHH_D_minus', 'rEHH_minus', 'EHH_A_plus', 'EHH_D_plus', 'rEHH_plus',
               'rEHH', 'IHH_A', 'IHH_D', 'iHS', 'mutation_age', 's', 'frequency']
    df_stats = pd.DataFrame(stats_list, columns=columns)
    df_stats.to_csv('{}/stats_datanum{}.csv'.format(stats_dir, params['data_num']))
    print('data num:', params['data_num'], 'done')


def main():
    # path to dir for mbs output
    data_dir = '../results'
    # path to dir for stats 
    stats_dir = '../data'
    # number of replication 
    n_run = 2
    # set number of cpu
    data_num = np.arange(1, 21, 1)
    # number of replication
    testlist = parameter_sets_forward(data_num)
    # window size in bp
    windowsize = 10000
    # ehh calculation pos in bp
    distance = 25000

    n_cpu = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(processes=n_cpu) as p:
        p.map(functools.partial(calc_EHH_sfs,
                                n_run=n_run, data_dir=data_dir, stats_dir=stats_dir,
                                windowsize=windowsize, distance = distance
                                ), testlist)


if __name__=="__main__":
    main()

