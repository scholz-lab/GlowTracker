## Running the website locally
1. Install [Ruby](https://www.ruby-lang.org/en/documentation/installation/)
2. Install __jekyll__
    ```bash
    gem install bundler jekyll
    ```
3. Goto `/docs` folder
4. Install the required Ruby packages
    ```bash
    bundle install
    ```
5. Run the website server
    ```bash
    bundle exec jekyll serve --livereload
    ```
6. The website can usually be accessed at [http://127.0.0.1:4000/](http://127.0.0.1:4000/)