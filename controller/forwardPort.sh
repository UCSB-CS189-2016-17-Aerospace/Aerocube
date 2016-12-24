iptables -t nat -A OUTPUT -p tcp --dport 950 -j REDIRECT --to-ports 5000
