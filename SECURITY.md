# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the repository maintainers directly
3. Include details about:
   - The vulnerability type
   - Steps to reproduce
   - Potential impact
   - Any mitigations you've identified

## Security Best Practices

### Environment Variables

- Never commit `.env` files with real credentials
- Use `.env.example` as a template
- Rotate credentials if they're accidentally exposed

### Dependencies

- Regularly update dependencies: `pip install --upgrade -r requirements.txt`
- Run security audits: `pip audit`

### Code Review

- All contributions are reviewed for security issues
- Avoid hardcoding secrets in code
- Use environment variables for sensitive data

## Known Security Considerations

- This project uses Selenium for NTU dataset downloads - ensure proper credential management
- Model checkpoints may be large - verify download sources
- No network services are exposed by default

## Security Updates

Security patches will be released as new versions. Check the changelog for security-related updates.
