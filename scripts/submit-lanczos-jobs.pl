#!/usr/bin/perl -w

use strict;
use warnings;
use FindBin qw($Script $Dir);
use Getopt::Long;
use KitaevUtils;

print "Starting\n";

# Available partitions:
# 'gpu' : ~16x Tesla K80, plus a bunch of Titan Black
# 'hns_gpu' : 8x Tesla K80
#
# mem_gb: this is the memory reported by CudaFactorizedHamiltonian.
#

my $run_params_N28 = {
    N => 28,
    min_run => 1,
    max_run => 200,
    secs_per_step => 0.04,
    mem_gb => 4,
    ev_steps => 20000,
};

my $run_params_N30 = {
    N => 30,
    min_run => 2,
    max_run => 50,
    secs_per_step => 0.04,
    mem_gb => 4,
    ev_steps => 20000,
};

my $run_params_N32 = {
    N => 32,
    min_run => 1,
    max_run => 50,
    secs_per_step => 0.04,
    mem_gb => 4,
    ev_steps => 20000,
};

my $run_params_N36 = {
    N => 36,
    min_run => 40,
    max_run => 50,
    secs_per_step => 0.12, # 0.0615
    mem_gb => 4,
    ev_steps => 20000,
};

my $run_params_N38 = {
    N => 38,
    min_run => 1,
    max_run => 50,
    secs_per_step => 0.23,
    mem_gb => 4,
};

my $run_params_N40 = {
    N => 40,
    min_run => 51,
    max_run => 200,
    secs_per_step => 1,
    mem_gb => 6, # 1.9
};

my $run_params_N42 = {
    N => 42,
#min_run => 51,
#max_run => 100,
    min_run => 35,
    max_run => 35,
    secs_per_step => 1.9,
    mem_gb => 8,
};

# N=42 with about 200 good eigenvalues instead of the usual 1000
my $run_params_N42_short = {
    prefix => 'shortlanc',
    N => 42,
    min_run => 95, max_run => 95,  
    secs_per_step => 1.9 * 1.2, # factor for flower GPUs
    mem_gb => 8,
    num_lanczos_steps => 12000,
};

my $run_params_N44 = {
    N => 44,
    min_run => 44,
    max_run => 50,
    secs_per_step => 5,
    mem_gb => 11, # reported 9.2
};

my $run_params_N46 = {
    N => 46,
    min_run => 51,
    max_run => 100,
    num_gpus => 4,
    secs_per_step => 20,
    mem_gb => 50,
    num_lanczos_steps => 50000,
};

my $run_params_N46_short = {
    prefix => 'shortlanc',
    N => 46,
    min_run => 1,
    max_run => 200,
    num_gpus => 4,
    secs_per_step => 20,
    mem_gb => 50,
    num_lanczos_steps => 12000,
};

my $run_params_N48 = {
    N => 48,
    min_run => 1,
    max_run => 1,
    num_gpus => 4,
    secs_per_step => 20, # TODO
    mem_gb => 70,
    num_lanczos_steps => 50000,
};

# !!! use submit-ground-job instead !!!
my $run_params_N32_ground_state = {
    prefix => 'gndlanc',
    N => 32,
    min_run => 401,
    max_run => 10000,
    secs_per_step => 0.04,
    mem_gb => 4,
    num_lanczos_steps => 300,
    checkpoint_steps => 300,
    ev_steps => 300,
    any_gpu => 1,
};

###### CHOOSE THE RUN ##################

#my $run_params = $run_params_N42;
#my $run_params = $run_params_N42_short;
my $run_params = $run_params_N46_short;

########################################

my $partitions = is_sherlock2() ? 'gpu' : 'gpu,hns_gpu';

if (!exists $run_params->{prefix}) {
    $run_params->{prefix} = "lanc";
}

if (!exists $run_params->{experiment}) {
    $run_params->{experiment} = "$run_params->{prefix}-N$run_params->{N}";
}

if (!exists $run_params->{checkpoint_dir}) {
    my $base_checkpoint_dir = KitaevUtils::data_dir() . "/lanczos-checkpoints";
    $run_params->{checkpoint_dir} = "$base_checkpoint_dir/$run_params->{experiment}";
}

if (!exists $run_params->{num_lanczos_steps}) {
    $run_params->{num_lanczos_steps} = 40000;
}

if (!exists $run_params->{checkpoint_steps}) {
    $run_params->{checkpoint_steps} = 200;
}

if (!exists $run_params->{ev_steps}) {
    $run_params->{ev_steps} = 2000;
}

my $debug = 0;
my $max_time_in_minutes = 2 * 24 * 60;
my $min_time_in_minutes = 2;
my $max_array_jobs = 100;

my $request_tesla;
my $request_gpu_model;

if (is_sherlock2()) {
    $request_tesla = 0;

    # On Sherlock 2, request P100 because P40 is slow on double precision
    $request_gpu_model = "GPU_SKU:TESLA_P100_PCIE";
}
else {
    $request_tesla = 1;
}

if (exists $run_params->{any_gpu} && $run_params->{any_gpu} == 1) {
    $request_tesla = 0;
    $request_gpu_model = undef;
}

my $data_dir = KitaevUtils::data_dir() . "/lanczos";
my $lanczos_prog = "./syk-gpu-lanczos";
my $lanczos_job_script = "jobs/lanczos-job";
my $J = 1;

mkdir $data_dir unless -d $data_dir;
#mkdir $base_checkpoint_dir unless -d $base_checkpoint_dir;
mkdir $run_params->{checkpoint_dir} unless -d $run_params->{checkpoint_dir};

die "Can't find checkpoint dir $run_params->{checkpoint_dir}" unless -d $run_params->{checkpoint_dir};

my $help = 0;

# Don't submit jobs, only print the command lines
my $dry_run = 0;

# Run locally instead of submitting the jobs
my $run_locally = 0;

my $run_num;

sub usage {
        print "Usage: $Script [--help] [-n] [--local] [--run 23]\n\n";
        print "If --run is provided, it overrides the default run range.\n\n";
}

GetOptions(
        'help' => \$help,
        'n' => \$dry_run,
        'local' => \$run_locally,
        'run=s' => \$run_num,
        );

if ($help) {
    usage();
    exit 0;
}

if (defined $run_num) {
    $run_params->{min_run} = $run_num;
    $run_params->{max_run} = $run_num;
}

print "calling submit_jobs\n";
submit_jobs($run_params);

sub submit_jobs {
    my ($run_params) = @_;

    my $N = $run_params->{N};
    my $num_gpus = 1;

    if (exists $run_params->{num_gpus}) {
        $num_gpus = $run_params->{num_gpus};
    }
    
    my $mem_mb = $run_params->{mem_gb} * 1024;
    my $time_minutes = int(
        $run_params->{secs_per_step} * $run_params->{num_lanczos_steps} / 60.);

    if ($num_gpus > 1) {
        # There is a 10% overhead when using multi-GPU configurations
        $time_minutes = $time_minutes / $num_gpus * 1.1;
    }

    if ($time_minutes > $max_time_in_minutes) {
        $time_minutes = $max_time_in_minutes;
    }
    elsif ($time_minutes < $min_time_in_minutes) {
        $time_minutes = $min_time_in_minutes;
    }
        
    print "N = $N:  memory = " . ($mem_mb/1024) . " GB,  time = "
        . ($time_minutes / 60) . " hours\n";

    my $run_name = "$run_params->{experiment}-run";

    my $lanczos_params = prepare_lanczos_params($run_name, $J, $run_params);

    my $min_run = $run_params->{min_run};
    my $max_run = $run_params->{max_run};

    while ($min_run <= $max_run) {
        my $num_jobs = $max_run - $min_run + 1;

        if ($num_jobs > $max_array_jobs) {
            $num_jobs = $max_array_jobs;
        }

        print "Calling submit_job_array\n";
        submit_job_array(
            $run_name, $min_run, $min_run + $num_jobs - 1,
            $lanczos_prog, $lanczos_params,
            $lanczos_job_script,
            $mem_mb, $time_minutes, $partitions, $num_gpus);

        $min_run += $num_jobs;
    }
}

sub prepare_lanczos_params {
    my ($run_name, $J, $run_params) = @_;

    my $lanczos_params = [
        "--N $run_params->{N}",
        "--J $J",
        "--data-dir $data_dir",
        "--checkpoint-dir $run_params->{checkpoint_dir}",
        "--num-steps $run_params->{num_lanczos_steps}",
        "--checkpoint-steps $run_params->{checkpoint_steps}",
        "--ev-steps $run_params->{ev_steps}",
        "--resume",
    ];

    if ($debug) {
        push @$lanczos_params, "--debug";
    }

    return $lanczos_params;
}

sub submit_job_array {
    my ($full_run_name, $min_run, $max_run,
        $prog, $params, $job_script,
        $job_mem_mb, $job_time_minutes, $partition, $num_gpus) = @_;

    if ($run_locally) {
        for my $run_idx ($min_run..$max_run) {
            my $cmd = "$prog --run-name ${full_run_name}${run_idx} " 
                . join(' ', @$params);
            KitaevUtils::execute($cmd, $dry_run);
        }

        return;
    }

    print "work_dir = " . KitaevUtils::work_dir() . "\n";
    $ENV{SGE_O_WORKDIR} = KitaevUtils::work_dir();
    $ENV{PARAM_FULL_RUN_NAME} = $full_run_name;
    $ENV{PARAM_PROG} = $prog;
    $ENV{PARAM_PROG_PARAMS} = join(' ', @$params);
    $ENV{PARAM_DATA_DIR} = $data_dir;

    print "PARAM_PROG_PARAMS = $ENV{PARAM_PROG_PARAMS}\n";

    if (! KitaevUtils::is_slurm_system()) {
        die "Can only submit Lanczos jobs on a SLURM system (Sherlock)";
    }

    my $out_file = "${data_dir}/${full_run_name}\%a.out";
    my $err_file = "${data_dir}/${full_run_name}\%a.err";

    # my $mem_opt = (defined $job_mem_mb ? "--mem=${job_mem_mb}" : "");
    # my $time_opt = (defined $job_time ? "--time=${job_time_minutes}" : "");
    my $mem_opt = "--mem=${job_mem_mb}";
    my $time_opt = "--time=${job_time_minutes}";

    my $sbatch_params = [
        "-J $full_run_name",
        "--array=${min_run}-${max_run}",
        "-o $out_file",
        "-e $err_file",
        "--export=ALL",
        "--mem-per-cpu=${job_mem_mb}M",
        "--time=$job_time_minutes",
        "--nodes=1",
        "-p $partition",
        ];

    if ($request_tesla) {
        push @$sbatch_params, "--gres gpu:tesla:$num_gpus";
    }
    else {
        push @$sbatch_params, "--gres gpu:$num_gpus";
    }

    if (defined $request_gpu_model) {
        push @$sbatch_params, "-C $request_gpu_model";
    }

    my $cmd = "sbatch " . join(' ', @$sbatch_params) . " $job_script";
    KitaevUtils::execute($cmd, $dry_run);
}

sub is_sherlock2 {
    die "Can't determine Sherlock version" unless exists $ENV{SHERLOCK};
    return $ENV{SHERLOCK} == 2;

#    my $host = `hostname`;
#    chomp $host;
#
#    if ($host =~ /^sherlock-/) {
#        return 0;
#    }
#    elsif ($host =~ /^sh-/) {
#        return 1;
#    }
#    else {
#        die "Cannot identify Sherlock cluster from hostname $host";
#    }
}
