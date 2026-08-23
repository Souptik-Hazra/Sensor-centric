const fs = require('fs');

const content = fs.readFileSync('../digital_twin_gis_map.html', 'utf-8');

const sensorsMatch = content.match(/const sensors = (\[[\s\S]*?\]);/);
if (sensorsMatch) {
  fs.writeFileSync('./public/sensors.json', sensorsMatch[1]);
  console.log('Extracted sensors.json');
}

const tsMatch = content.match(/const tsData = (\{[\s\S]*?\});\s+const hasEmpiricalData/);
if (tsMatch) {
  fs.writeFileSync('./public/traffic_data.json', tsMatch[1]);
  console.log('Extracted traffic_data.json');
}
