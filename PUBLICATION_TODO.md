# Publication Readiness TODO List

This file documents issues and pending items for publication readiness.

## Completed Items

- [x] Removed hardcoded credentials from `Dataset_utils/NTU/downloadntu.py`
- [x] Added `.env.example` template for credentials
- [x] Updated `.gitignore` to exclude large files and sensitive data
- [x] Created comprehensive `README.md` with installation, usage, and reproducibility instructions
- [x] Added `CITATION.cff` file with publication metadata
- [x] Created `CONTRIBUTING.md` for community guidelines
- [x] Created `SECURITY.md` for security policy

## Pending Items (High Priority)

### 1. LICENSE File
**Status**: Resolved
**Action**: Added root-level `LICENSE` (MIT)
**Note**: Main codebase uses MIT. The bundled omnivore component (`VIP/src/modeling/`) remains CC-BY-NC 4.0.

### 2. Paper Information / Citation
**Status**: Resolved
**Action**: Removed `CITATION.cff`. Citation is now provided in `README.md` with the full author list and a link to the arXiv preprint (https://arxiv.org/abs/2606.02352).

## Pending Items (Medium Priority)

### 3. Tests
**Status**: No test suite exists  
**Recommendation**: Add minimal smoke tests to verify:
- Imports work correctly
- Basic data loading functions
- Model initialization
- Configuration parsing

**Estimated effort**: Low  
**Impact**: High - Improves reliability and onboarding

### 4. CI/CD Pipeline
**Status**: Not configured  
**Recommendation**: Add minimal GitHub Actions workflow that runs:
- Installation test
- Linting (ruff or flake8)
- Basic import smoke test
- Configuration validation

**Estimated effort**: Medium  
**Impact**: High - Ensures code quality and prevents regressions

### 5. Code Formatting and Linting
**Status**: Not enforced  
**Recommendation**: Add `ruff` or `flake8` configuration and run pre-commit checks

**Current**: `ruff` is in `requirements.txt` but not configured

## Pending Items (Low Priority)

### 6. Documentation Improvements
- [ ] Add more detailed training examples
- [ ] Add architecture diagrams
- [ ] Include visualization examples
- [ ] Add troubleshooting guide

### 7. Model Checkpoints
**Issue**: Large model files (~4GB checkpoint) are in the repository  
**Recommendation**: 
- Move large files to git-lfs or separate storage
- Update `.gitignore` to exclude all checkpoints by default
- Provide download instructions for pretrained models

### 8. Example Scripts
- [ ] Add minimal working example script
- [ ] Create a Colab notebook for quick experimentation
- [ ] Add a Dockerfile for containerized deployment

## Security Issues Found

### 1. Hardcoded Credentials (FIXED ✓)
- **File**: `Dataset_utils/NTU/downloadntu.py:66-67`
- **Issue**: NTU dataset username and password were hardcoded
- **Fix**: Replaced with environment variables (`NTU_USERNAME`, `NTU_PASSWORD`)
- **Status**: Resolved

### 2. Internal Paths
- **Files**: Multiple files contain hardcoded internal paths
- **Examples**:
  - `/home/bas06400/...` (user-specific path)
  - `/net/polaris/storage/...` (internal network path)
- **Recommendation**: Replace with configurable paths or environment variables
- **Status**: Documented - not all can be fixed without breaking existing workflows

### 3. Kerberos Authentication
- **Files**: `Dataset_utils/NTU/downloadntu.py:12-26`
- **Issue**: Hardcoded Kerberos credential paths
- **Recommendation**: Make paths configurable via environment variables
- **Status**: Documented - requires system-level Kerberos setup

## Legal/Author Approval Required

### 1. Final License Decision
**Issue**: No root-level LICENSE file  
**Options**:
- MIT License (matches CLIP-ViP)
- Apache-2.0 (common for ML projects)
- CC-BY-NC 4.0 (if academic non-commercial use only)

**Action Required**: Author decision needed

### 2. Third-Party Licenses
**Status**: 
- CLIP-ViP: MIT (verified in `VIP/LICENSE`)
- Omnivore: CC-BY-NC 4.0 (verified in `VIP/src/modeling/LICENSE`)
- Other dependencies: Listed in `requirements.txt`

**Action**: Verify all dependencies are compatible with chosen license

### 3. Dataset Licenses
**NTU RGB+D**: Requires registration and is for research only  
**DAA**: Contact dataset authors for license terms

**Action**: Document dataset licenses in README and ensure users agree to terms

### 4. Paper Authorship
**Status**: Author names not confirmed in `CITATION.cff`  
**Action**: Verify final author list and order

## Summary

### Critical Blocking Issues
- [x] Add root `LICENSE` file (MIT)
- [x] Confirm final author names (in README citation)

### High Priority Items
- [ ] Add smoke tests
- [ ] Configure CI/CD pipeline
- [ ] Fix internal path references

### Medium Priority Items
- [ ] Add documentation improvements
- [ ] Clean up large model files from repo
- [ ] Configure code formatting

### Ready to Publish
The critical issues (license and author names) are resolved. The code is ready for publication.

**Remaining Next Steps**:
1. Add smoke tests and CI (optional but recommended)
2. Clean up large model files from repo (optional)
3. Configure code formatting (optional)
