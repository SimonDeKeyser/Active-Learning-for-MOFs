import paramiko
import re
import logging
logging.basicConfig(format='',level=logging.INFO)

class VSC_shell:
    def __init__(self, host, user, psw, key):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(host, username=user, password=psw, key_filename=key)

        channel = self.ssh.invoke_shell()
        self.stdin = channel.makefile('wb')
        self.stdout = channel.makefile('r')

    def __del__(self):
        self.ssh.close()

    def execute(self, cmd):
        """

        :param cmd: the command to be executed on the remote computer
        :examples:  execute('ls')
                    execute('finger')
                    execute('cd folder_name')
        """
        cmd = cmd.strip('\n')
        self.stdin.write(cmd + '\n')
        finish = 'end of stdOUT buffer. finished with exit status'
        echo_cmd = 'echo {} $?'.format(finish)
        self.stdin.write(echo_cmd + '\n')
        shin = self.stdin
        self.stdin.flush()

        shout = []
        sherr = []
        exit_status = 0
        for line in self.stdout:
            if str(line).startswith(cmd) or str(line).startswith(echo_cmd):
                # up for now filled with shell junk from stdin
                shout = []
            elif str(line).startswith(finish):
                # our finish command ends with the exit status
                exit_status = int(str(line).rsplit(maxsplit=1)[1])
                if exit_status:
                    # stderr is combined with stdout.
                    # thus, swap sherr with shout in a case of failure.
                    sherr = shout
                    shout = []
                break
            else:
                # get rid of 'coloring and formatting' special characters
                shout.append(re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', line).
                             replace('\b', '').replace('\r', ''))

        # first and last lines of shout/sherr contain a prompt
        if shout and echo_cmd in shout[-1]:
            shout.pop()
        if shout and cmd in shout[0]:
            shout.pop(0)
        if sherr and echo_cmd in sherr[-1]:
            sherr.pop()
        if sherr and cmd in sherr[0]:
            sherr.pop(0)

        return shin, shout, sherr

    def swap_cluster(self, cluster):
        _, _, _ = self.execute('ssh login8')
        _, _, _ = self.execute('module --force purge')
        _, _, _ = self.execute('module load cluster/{}'.format(cluster))
        _, _, _ = self.execute('module swap cluster/{}'.format(cluster))
        _, shout, sherr = self.execute('module list')

        logging.info('Sucesfully swapped to {}'.format(shout[2]))

    def submit_job(self, cluster, rundir, jobfile):
        self.swap_cluster(cluster)
        _, _, _ = self.execute('cd {}'.format(rundir))
        _, shout, sherr = self.execute('qsub {} -d $(pwd)'.format(jobfile))

        if sherr == []:
            logging.info('Successfully submitted job: {}'.format(shout[0]))
        else:
            logging.info(sherr)

    def start_sh(self, cluster, rundir, jobfile):
        self.swap_cluster(cluster)
        _, _, _ = self.execute('cd {}'.format(rundir))
        _, shout, sherr = self.execute('bash {}'.format(jobfile))

        if sherr == []:
            logging.info('Successfully started .sh: {}'.format(shout[0]))
        else:
            logging.info(sherr)