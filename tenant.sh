#!/usr/bin/env bash
# tenant.sh — human control script for repaveable bawthub/llm-bawt tenant VMs.
# Run ON echo as nick.  See the `tenant-provisioning` skill for the full rationale.
#
#   ./tenant.sh new [name] [fwd_port]   provision a brand-new instance end-to-end
#   ./tenant.sh up      [name]          start stack + LAN forward (existing VM)
#   ./tenant.sh stop    [name]          stop stack + LAN forward (VM keeps running)
#   ./tenant.sh restart [name]          stop then up
#   ./tenant.sh status  [name]          VM state, containers, forward, URL
#   ./tenant.sh url     [name]          print the click-in URL
#   ./tenant.sh logs    [name] [svc]    tail stack logs
#   ./tenant.sh seed    [name]          apply the idempotent infrastructure seed
#                                       (schema, access paths, prompt defaults — no bots;
#                                       the wizard creates the first bot after a provider connects)
#   GitHub auth: connect in setup/settings; the supported bootstrap installer
#   configures the image-baked helper automatically.
#   ./tenant.sh destroy [name]          DELETE the VM + disk + forward (irreversible)
#
# Defaults: name=bawthub-inst1  fwd_port=8830  (forward lives on echo -> VM:3000)
set -euo pipefail

NAME="${2:-bawthub-inst1}"
FWD_PORT="${3:-8830}"
FWD_UNIT="${NAME}-fwd"
LAN_IP="10.0.0.101"                       # echo's LAN address (where humans browse)
IMG_DIR="/var/lib/libvirt/images"
BASE_IMG="${IMG_DIR}/noble-server-cloudimg-amd64.img"
COMPOSE="$(cd "$(dirname "$0")" && pwd)/docker-compose.prod.yml"
VCPUS=6; MEM=12288; DISK=80G
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=6"

vm_ip() { sudo virsh net-dhcp-leases default 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1; }
vm_exists() { sudo virsh dominfo "$NAME" >/dev/null 2>&1; }

cmd_new() {
  vm_exists && { echo "VM '$NAME' already exists — use 'up' or 'destroy' first."; exit 1; }
  [ -f "$BASE_IMG" ] || sudo curl -fsSL -o "$BASE_IMG" \
    https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img
  echo "==> disk"
  sudo cp "$BASE_IMG" "${IMG_DIR}/${NAME}.qcow2"
  sudo qemu-img resize "${IMG_DIR}/${NAME}.qcow2" "$DISK"
  echo "==> cloud-init seed (cloud-localds — do NOT use virt-install --cloud-init)"
  local PUB; PUB="$(cat ~/.ssh/id_rsa.pub)"
  sudo tee "${IMG_DIR}/${NAME}-user-data" >/dev/null <<EOF
#cloud-config
hostname: ${NAME}
manage_etc_hosts: true
users:
  - name: bawthub
    groups: [sudo]
    shell: /bin/bash
    sudo: 'ALL=(ALL) NOPASSWD:ALL'
    ssh_authorized_keys: ["${PUB}"]
package_update: true
packages: [ca-certificates, curl]
runcmd:
  - curl -fsSL https://get.docker.com | sh
  - systemctl enable --now docker
  - usermod -aG docker bawthub
  - [ touch, /var/lib/cloud-init-done ]
EOF
  printf 'instance-id: %s-001\nlocal-hostname: %s\n' "$NAME" "$NAME" \
    | sudo tee "${IMG_DIR}/${NAME}-meta-data" >/dev/null
  sudo cloud-localds "${IMG_DIR}/${NAME}-seed.iso" \
    "${IMG_DIR}/${NAME}-user-data" "${IMG_DIR}/${NAME}-meta-data"
  echo "==> virt-install"
  sudo virt-install --name "$NAME" --memory "$MEM" --vcpus "$VCPUS" --cpu host-passthrough \
    --disk "path=${IMG_DIR}/${NAME}.qcow2,format=qcow2,bus=virtio" \
    --disk "path=${IMG_DIR}/${NAME}-seed.iso,device=cdrom" \
    --os-variant ubuntu24.04 --network network=default,model=virtio \
    --graphics none --noautoconsole --import
  echo "==> waiting for cloud-init (docker install)…"
  local IP=""; for _ in $(seq 1 60); do
    IP="$(vm_ip)"; [ -n "$IP" ] && $SSH "bawthub@$IP" 'test -f /var/lib/cloud-init-done' 2>/dev/null && break
    sleep 6
  done
  [ -n "$IP" ] || { echo "VM never became reachable"; exit 1; }
  echo "==> deploying stack to $IP"
  $SSH "bawthub@$IP" "echo \$(gh auth token 2>/dev/null || true) | docker login ghcr.io -u zenoran --password-stdin" 2>/dev/null \
    || echo "  (VM ghcr login skipped — pass a read token if images are private)"
  scp -o StrictHostKeyChecking=no "$COMPOSE" "bawthub@$IP:~/docker-compose.prod.yml" >/dev/null
  # .env: DB password + the provider-credential encryption key. LLM_BAWT_SECRET_KEY
  # MUST persist across repaves or stored provider credentials (GitHub token, LLM
  # API keys) become unreadable — so it lives in the VM's .env, generated once.
  $SSH "bawthub@$IP" "test -f .env || { \
      echo LLM_BAWT_POSTGRES_PASSWORD=\$(openssl rand -hex 24); \
      echo LLM_BAWT_SECRET_KEY=\$(python3 -c 'import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())'); \
    } > .env; \
    docker compose -f docker-compose.prod.yml pull && docker compose -f docker-compose.prod.yml up -d"
  start_forward "$IP"
  echo "==> DONE  ->  http://${LAN_IP}:${FWD_PORT}"
}

start_forward() {
  local IP="${1:-$(vm_ip)}"
  sudo systemctl stop "$FWD_UNIT" 2>/dev/null || true
  sudo systemctl reset-failed "$FWD_UNIT" 2>/dev/null || true
  sudo systemd-run --unit="$FWD_UNIT" --collect \
    /usr/bin/socat "TCP-LISTEN:${FWD_PORT},fork,reuseaddr" "TCP:${IP}:3000"
}

# Compatibility notice only: never overwrite the image's current helper.
cmd_bridge_git_setup() {
  echo "bridge-git-setup retired: use bawthub/deploy/bootstrap.sh and connect GitHub in setup/settings." >&2
  return 1
}

cmd_up()   { local IP; IP="$(vm_ip)"; $SSH "bawthub@$IP" 'docker compose -f docker-compose.prod.yml up -d'; start_forward "$IP"; echo "http://${LAN_IP}:${FWD_PORT}"; }
cmd_stop() { local IP; IP="$(vm_ip)"; $SSH "bawthub@$IP" 'docker compose -f docker-compose.prod.yml down' || true; sudo systemctl stop "$FWD_UNIT" 2>/dev/null || true; echo "stopped."; }
cmd_restart() { cmd_stop; cmd_up; }
cmd_url()  { echo "http://${LAN_IP}:${FWD_PORT}"; }
cmd_logs() {
  local IP service command="docker compose -f docker-compose.prod.yml logs -f --tail=100"
  IP="$(vm_ip)"
  if [ -n "${1:-}" ]; then
    printf -v service '%q' "$1"
    command+=" $service"
  fi
  $SSH "bawthub@$IP" "$command"
}
cmd_seed() {
  local IP; IP="$(vm_ip)"
  [ -n "$IP" ] || { echo "VM not reachable"; exit 1; }
  echo "==> applying idempotent tenant infrastructure seed (no bots)"
  $SSH "bawthub@$IP" 'docker compose -f docker-compose.prod.yml exec -T app python -m llm_bawt.seeding'
}
cmd_status() {
  echo "VM $NAME: $(sudo virsh domstate "$NAME" 2>/dev/null || echo 'does not exist')"
  local IP; IP="$(vm_ip)"; echo "IP: ${IP:-none}"
  [ -n "$IP" ] && $SSH "bawthub@$IP" 'docker compose -f docker-compose.prod.yml ps' 2>/dev/null || true
  echo "forward $FWD_UNIT: $(sudo systemctl is-active "$FWD_UNIT" 2>/dev/null || echo inactive)  ->  http://${LAN_IP}:${FWD_PORT}"
}
cmd_destroy() {
  read -rp "DELETE VM '$NAME' + its disk permanently? [y/N] " a; [ "$a" = y ] || { echo aborted; exit 0; }
  sudo systemctl stop "$FWD_UNIT" 2>/dev/null || true
  sudo virsh destroy "$NAME" 2>/dev/null || true
  sudo virsh undefine "$NAME" --nvram --remove-all-storage 2>/dev/null || true
  sudo rm -f "${IMG_DIR}/${NAME}-seed.iso" "${IMG_DIR}/${NAME}-user-data" "${IMG_DIR}/${NAME}-meta-data"
  echo "destroyed."
}

case "${1:-status}" in
  new) cmd_new ;; up) cmd_up ;; stop) cmd_stop ;; restart) cmd_restart ;;
  status) cmd_status ;; url) cmd_url ;; logs) cmd_logs "${3:-}" ;; seed) cmd_seed ;; destroy) cmd_destroy ;;
  bridge-git-setup) cmd_bridge_git_setup ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20 ;;
esac
