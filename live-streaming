#!/bin/bash

simulate=false
workdir=/mnt/ramdisk/segmented
vbase=video
abase=bee-audio
mp4base=new-video

m3ufile=play-list.m3u8

# for each one video segment, we create a m3u8 file: ts-list.m3u8,
# then we move it to ts-list-$sq.m3u8. We keep the latest 10 m3u8
# files so we can easily combine them into play-list.m3u8
segbase=ts-list

maxseq=300
maxduration=10
# duration of a clip (?)
duration=10
durationms=10000

adevice=hw:3,0

capturepidfile=$workdir/capture.pid
combinepidfile=$workdir/combine.pid

samplen=1
sampledir=${workdir}/samples

# sample_file <file> <sq>
function sample_file() {
    local fname=$1
    local sq=$2

    if [  "$samplen" -gt 0 ] && [ $(( $sq % $samplen ))  -eq 0 ]; then
        echo "samplen: $samplen, sq: $sq, !!!!!!!!===== doing sampling  =====!!!!!!!!! "
        cp $fname $sampledir
    fi
}

# returns true if the job is already running
function check_already_running() {
    local pidfile=$1
  
    if [ -f $pidfile ]; then
        local pid=`cat $pidfile`
        ps ax | awk '{print $1}' &> $pidfile.tmp
        grep $pid $pidfile.tmp &> /dev/null
        found=$?
        rm $pidfile.tmp

        if [ $found -eq 0 ]; then
            return 1  # pid found, process is running
        fi
    fi

    # not already running, write pidfile, and return 0
    echo "writing pid: $$ to file: $pidfile" 
    echo $$ > $pidfile
}

function wait_for_file() {
    local fname=$1
    until [ -f $fname ] ; do
        sleep 0.05
    done
}

# takes in one parameter, the sequence
function capture_one() { 
    local sq=$1

    afile=$abase-$sq.wav
    vfile=$vbase-$sq.h264

    rm -f $afile $vfile

    if $simulate; then
        sleep $duration
        echo "test" > $afile
        echo "test" > $vfile
    else
        libcamera-vid --autofocus-mode continuous -t $durationms --width 1920 --height 1080 -o $vfile.tmp && mv $vfile.tmp $vfile &
        arecord -D $adevice -d $duration -f cd  $afile.tmp && mv $afile.tmp $afile &
    fi

    wait_for_file $afile
    wait_for_file $vfile

    echo " == afile: $afile, vfile: $vfile both created"
}

function capture() {
    local num=$1
    check_already_running $capturepidfile
    if [ $? -eq 1 ]; then
        echo "Capture loop already running, exiting.."
        return 0
    fi
    for i in `seq 1 $num`; do 
        echo "==== Going to capture video+audio for sequence $i"
        capture_one $i
        if [ $duration -lt $maxduration ]; then
            duration=$(( $duration + 1 ))
            durationms=$(( $durationms + 1000 ))
        fi
    done
    rm $capturepidfile
}

function update_play_list() {
    local sq=$1

    start=1
    if [ $sq -gt 10 ]; then
        start=$(( $sq - 10 ))
    fi

    cat <<EOT > $m3ufile.tmp
#EXTM3U 
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
EOT

    cat $segbase-$start.m3u8 | head -n 6 | tail -n 3 >> $m3ufile.tmp

    while [ $start -lt $sq ]; do
        start=$(( $start + 1 ))
        cat $segbase-$start.m3u8 | tail -n 3 | head -n 2 >> $m3ufile.tmp
    done

    mv $m3ufile.tmp $m3ufile
}

# ts-list$sq.mp4, ts-list-sq.m3u8
function clean_up_files() {
    local sq=$1

    if [ $sq -gt 10 ]; then
        start=$(( $sq - 10 ))
        vseg=${segbase}${start}.ts
        vlist=$segbase-$start.m3u8
        echo "______removing $vseg $vlist for gc"
        rm $vseg $vlist
    fi
}

function combine_one() {
    local sq=$1

    afile=$abase-$sq.wav
    vfile=$vbase-$sq.h264

    wait_for_file $afile
    wait_for_file $vfile

    echo "ready to combine afile and vfile for seq: $sq"

    output="$mp4base-$sq.mp4"

    if $simulate; then
        sleep $(( $duration - 1 ))
        echo "test" > $output
        echo "test" > ${segbase}${sq}.ts
        cp test-list-template.m3u8 $segbase.m3u8
    else
        ffmpeg -i $vfile -i $afile -map 0:v -map 1  -c:v copy -c:a aac $output
        ffmpeg -i $output -codec: copy -bsf:v h264_mp4toannexb -start_number $sq \
               -hls_time 12 -hls_list_size 0 -f hls $segbase.m3u8
    fi

    mv $segbase.m3u8 $segbase-$sq.m3u8

    # uncomment the following line if we want to sample the video files.
    # sample_file $output $sq

    sample_file $afile $sq

    rm $afile $vfile $output
}

function combine() {
    num=$1
    check_already_running $combinepidfile
    if [ $? -eq 1 ]; then
        echo "-- Combine loop already running, exiting.."
        return 0
    fi

    for i in `seq 1 $num`; do 
        echo "--- Going to combine video+audio for sequence $i"
        combine_one $i

        update_play_list $i

        # ts-list$sq.mp4, ts-list-$sq.m3u8
        clean_up_files $i
    done
    echo "#EXT-X-ENDLIST" >> $m3ufile
    rm $combinepidfile
}

cd $workdir

echo "command $0 running, num params: $#, cur_dir: `pwd`"

acard=`aplay -l | grep Audio | tr ':' ' ' | awk '{print $2}'`
adevice="hw:${acard},0"

if [ "$adevice" != "hw:1,0" ] && [ "$adevice" != "hw:3,0" ] && [ "$adevice" != "hw:2,0" ]; then
    echo "Failed to get audio card. Got '$adevice'".
    exit 1
else
    echo "Using audio device: '$adevice'"
fi


if [ $# -eq 1 ]; then
    num=$1
    if [ -n "$num" ] && [ "$num" -eq "$num" ] 2>/dev/null; then
        if [ $num -gt $maxseq ]; then
            num=$maxseq
        fi
        echo "running whole thing for $num segments..."
    else
        echo "Only one argument provided, but not a number! arg is: '$1', exit..."
        exit 1
    fi

    if [ -f $m3ufile ]; then
        rm $m3ufile
    fi
    $0 capture $num &
    $0 combine $num &
    wait_for_file $m3ufile
    echo "~~~~~ $m3ufile created!!~~~~"
elif [ $# -eq 2 ]; then
    cmd=$1
    num=$2
    echo "cmd: $cmd, num: $num"

    if [ "$cmd" = "capture" ]; then
        echo "start capturing loop, num is $num..."
        capture $num
    elif [ "$cmd" = "combine" ]; then
        echo "start combining loop, num is $num..."
        combine $num
    elif [ "$cmd" = "capture_one" ]; then
        echo "==== test capture_one ==="
        capture_one 1
    elif [ "$cmd" = "combine_one" ]; then
        echo "==== test combine_one ==="
        combine_one 1
    else
        echo "unknown command: '$cmd', num: '$num'"
    fi
fi