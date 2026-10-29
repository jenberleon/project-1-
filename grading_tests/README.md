
When testing a GitHub link, the tests download the repository into `grading_tests/github_workdir`. This folder must not exist when the tests start, so delete it before running them again:

```bash
cd grading_tests
rm -rf github_workdir
```

The folder is listed in `.gitignore`, so it is never uploaded to GitHub.
