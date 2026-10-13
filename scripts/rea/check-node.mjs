// Match rea-agents 6.4.0's declared engine; do not install another runtime.
const [major, minor] = process.versions.node.split('.').map(Number);
if (!((major === 22 && minor >= 19) || (major === 24 && minor >= 11) || major >= 26)) {
  process.stderr.write(`REA requires Node ^22.19.0 || ^24.11.0 || >=26.0.0; found ${process.versions.node}\n`);
  process.exit(1);
}
