#!/usr/bin/env node

/**
 * LiteClaw — Version Bumper
 *
 * Usage:
 *   node scripts/bump-version.js <new-version>
 *   node scripts/bump-version.js [patch|minor|major]
 *
 * Example:
 *   node scripts/bump-version.js 1.0.2
 *   node scripts/bump-version.js patch
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const rootDir = join(dirname(__filename), '..');

const pkgPath = join(rootDir, 'package.json');
const lockPath = join(rootDir, 'package-lock.json');

if (!existsSync(pkgPath)) {
  console.error('Error: package.json not found in ' + rootDir);
  process.exit(1);
}

const pkg = JSON.parse(readFileSync(pkgPath, 'utf-8'));
const currentVersion = pkg.version || '1.0.0';

const targetArg = process.argv[2];

if (!targetArg) {
  console.log(`Current LiteClaw version: ${currentVersion}`);
  console.log('\nUsage:');
  console.log('  npm run bump <version>       (e.g. npm run bump 1.0.2)');
  console.log('  npm run bump [patch|minor|major]');
  process.exit(0);
}

function calculateNextVersion(current, bumpType) {
  const parts = current.split('.').map(Number);
  if (parts.length < 3 || parts.some(isNaN)) {
    throw new Error(`Invalid current version format: ${current}`);
  }
  let [major, minor, patch] = parts;
  if (bumpType === 'major') {
    major += 1;
    minor = 0;
    patch = 0;
  } else if (bumpType === 'minor') {
    minor += 1;
    patch = 0;
  } else if (bumpType === 'patch') {
    patch += 1;
  } else {
    throw new Error(`Unknown bump type: ${bumpType}`);
  }
  return `${major}.${minor}.${patch}`;
}

let nextVersion = targetArg;
if (['patch', 'minor', 'major'].includes(targetArg.toLowerCase())) {
  nextVersion = calculateNextVersion(currentVersion, targetArg.toLowerCase());
} else {
  // Validate semver format
  if (!/^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/.test(targetArg)) {
    console.error(`Error: "${targetArg}" is not a valid semver version.`);
    process.exit(1);
  }
}

if (nextVersion === currentVersion) {
  console.log(`Version is already ${nextVersion}. No changes needed.`);
  process.exit(0);
}

// 1. Update package.json
pkg.version = nextVersion;
writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + '\n', 'utf-8');
console.log(`✓ Updated package.json: ${currentVersion} -> ${nextVersion}`);

// 2. Update package-lock.json if present
if (existsSync(lockPath)) {
  try {
    const lock = JSON.parse(readFileSync(lockPath, 'utf-8'));
    lock.version = nextVersion;
    if (lock.packages && lock.packages['']) {
      lock.packages[''].version = nextVersion;
    }
    writeFileSync(lockPath, JSON.stringify(lock, null, 2) + '\n', 'utf-8');
    console.log(`✓ Updated package-lock.json: ${currentVersion} -> ${nextVersion}`);
  } catch (err) {
    console.warn(`! Failed to update package-lock.json: ${err.message}`);
  }
}

console.log(`\n🎉 LiteClaw bumped to v${nextVersion}!`);
console.log('Next steps:');
console.log('  1. Update CHANGELOG section in README.md if needed.');
console.log('  2. Run: npm run build');
console.log(`  3. Commit and push: git commit -am "chore: bump version to ${nextVersion}"`);
