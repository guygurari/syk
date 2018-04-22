package KitaevUtils;

use strict;
use warnings;
use FindBin qw($Script $Dir);
use IO::File;
use Cwd;

my $job_script = 'jobs/kitaev-job';
my $max_jobs = 480;

# Resolve data_dir to its full path, which is on /farmshare/user_data/...
# This is important because when the SSH session ends, eventually the
# jobs lose permission to write to the homedir. So we want all writes
# to go directory to the resolved data dir.
my $work_dir = Cwd::realpath($Dir);
my $data_dir = Cwd::realpath("$Dir/data");
die "Missing data dir '$data_dir'" unless -d $data_dir;

sub data_dir {
    return $data_dir;
}

sub work_dir {
    return $work_dir;
}

sub execute {
    my ($cmd, $dry_run) = @_;
    print ("\n$cmd\n");

    if (!defined $dry_run || !$dry_run) {
        system($cmd);
        die $@ if $? >> 8;
    }
}

sub is_slurm_system {
    system("which sbatch > /dev/null");
    my $rc = $? >> 8;
    return $rc == 0;
}

sub is_qsub_system {
    system("which qsub > /dev/null");
    my $rc = $? >> 8;
    return $rc == 0;
}

sub submit_job {
    my ($full_run_name, 
        $dry_run,
        $run_locally,
        $prog, $params,
        $prog2, $params2,
        $job_mem_mb, $job_time,
    ) = @_;

    if ($run_locally) {
        my $cmd = "$prog " . join(' ', @$params);
        execute($cmd, $dry_run);

        if (defined $prog2) {
            my $cmd2 = "$prog2 " . join(' ', @$params2);
            execute($cmd2, $dry_run);
        }
    }
    else {
        $ENV{SGE_O_WORKDIR} = $work_dir;
        $ENV{PARAM_FULL_RUN_NAME} = $full_run_name;
        $ENV{PARAM_PROG} = $prog;
        $ENV{PARAM_PROG_PARAMS} = join(' ', @$params);

        if (defined $prog2) {
            $ENV{PARAM_PROG2} = $prog2;
            $ENV{PARAM_PROG2_PARAMS} = join(' ', @$params2);
        }

        if (is_slurm_system()) {
            my $out_file = "${data_dir}/${full_run_name}.out";
            my $err_file = "${data_dir}/${full_run_name}.err";

            my $mem_opt = (defined $job_mem_mb ? "--mem=${job_mem_mb}" : "");
            my $time_opt = (defined $job_time ? "--time=${job_time}" : "");

            my $cmd = "sbatch -J ${full_run_name} -o $out_file -e $err_file --export=ALL $mem_opt $time_opt $job_script";
            execute($cmd, $dry_run);
        }
        elsif (is_qsub_system()) {
            my $cmd = "qsub -N ${full_run_name} -V -o $data_dir -e $data_dir $job_script";
            execute($cmd, $dry_run);
        }
        else {
            die "Don't know how to submit jobs";
        }
    }
}

# Checks whether there is room to submit at least one job
sub can_submit_jobs {
    my $qstat = IO::File->new("qstat|");

    <$qstat>;
    <$qstat>;

    my $total = 0;
    my $running = 0;

    while (<$qstat>) {
        if (/  r  /) {
            $running++;
        }
        $total++;
    }

    print "(there are $total jobs in the queue)\n";
    return $total < $max_jobs;
}

1;

