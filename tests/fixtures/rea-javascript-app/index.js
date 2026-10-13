import { writeFileSync } from 'node:fs';
import { describeFeature } from './feature.js';

// Static analysis must NEVER execute this module. A smoke test checks absence.
writeFileSync(new URL('./EXECUTED', import.meta.url), 'Fixture was executed!');
export const description = describeFeature('static-smoke');
