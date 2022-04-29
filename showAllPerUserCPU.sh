set -e

own=$(id -nu)
cpus=$(lscpu | grep "^CPU(s):" | awk '{print $2}')

for user in $(getent passwd | awk -F ":" '{print $1}' | sort -u)
do
    # print other user's CPU usage in parallel but skip own one because
    # spawning many processes will increase our CPU usage significantly
    if [ "$user" = "$own" ]; then continue; fi
    (top -b -n 1 -u "$user" | awk -v user=$user -v CPUS=$cpus 'NR>7 { sum += $9; } END { if (sum > 0.0) print user, sum; }') &
    # don't spawn too many processes in parallel
    sleep 0.05
done
wait

# print own CPU usage after all spawned processes completed
top -b -n 1 -u "$own" | awk -v user=$own -v CPUS=$cpus 'NR>7 { sum += $9; } END { print user, sum; }'

print 1111
