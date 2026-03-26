def Pipeline(String DockerImg, String DockerRunArgs, String GpuType) {
    docker.image("${DockerImg}").inside("${DockerRunArgs}") {
        sh '''
            set -e
            rsync -a --delete --exclude 'Paddle/' --exclude 'backends/musa/third_party/' /workspace/ /home/paddle_musa/
        '''

        gitlabCommitStatus(name: "01-${GpuType}-env prepare", state: "running") {
            sh '''
                set -e
                cd /home/paddle_musa

                apt-get update && apt-get install -y patchelf

                pip3 install -r ./Paddle/python/requirements.txt
  
                pip3 install -r ./python/requirements.txt
            '''
        }

        gitlabCommitStatus(name: "02-${GpuType}-build paddle", state: "running") {
            sh '''
                set -e
                cd /home/paddle_musa/backends/musa
                echo y | bash tools/build.sh -p

                pip3 install numpy==1.22.4

                echo y | bash tools/build.sh -m
            '''
        }

        gitlabCommitStatus(name: "03-${GpuType}-unit test", state: "running") {
            sh '''
                set -e
                cd /home/paddle_musa
                export PADDLE_XCCL_BACKEND=musa
                export MUSA_VISIBLE_DEVICES=6,7
                cd ./backends/musa && bash tools/run_ut.sh
            '''
        }
    }
}

pipeline {
  agent none

  options {
    gitLabConnection('sh-code')
  }

  environment {
    S5000IMG = 'sh-harbor.mthreads.com/mt-ai/musa-paddle-dev:20260114_434'
    DOCKER_RUN_ARGS = '--network=host ' +
      '--user root ' +
      '--privileged ' +
      '--shm-size 20G ' +
      '--pid=host ' +
      '-e MTHREADS_VISIBLE_DEVICES=all ' +
      '-e MUSA_VISIBLE_DEVICES=all ' +
      '-v $WORKSPACE:/workspace'
  }

  stages {
    stage('Run task in parallel') {
      parallel {
        stage('paddle_musa') {
          agent { label 'paddle_musa' }
          steps {
            deleteDir()
            checkout scm
            timeout(time: 200, unit: 'MINUTES') {
              script {
                Pipeline("${S5000IMG}", "${DOCKER_RUN_ARGS}", "S5000")
              }
            }
          }
        }
      }
    }
  }

  post {
    unstable {
      script {
        currentBuild.result = 'FAILURE'
        error("Build marked as FAILURE due to instability.")
      }
      updateGitlabCommitStatus name: '06-final', state: 'failed'
    }
    failure {
      updateGitlabCommitStatus name: '06-final', state: 'failed'
    }
    success {
      updateGitlabCommitStatus name: '06-final', state: 'success'
    }
    aborted {
      updateGitlabCommitStatus name: '06-final', state: 'canceled'
    }
  }
}
