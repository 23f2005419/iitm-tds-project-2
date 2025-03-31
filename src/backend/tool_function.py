tool_functions = [
        {
            "type": "function",
            "function": {
                "name": "execute_commands",
                "description": "Execute a shell command and return its output. Use this for questions about running terminal commands like 'code -s' to check VS Code status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute (e.g., 'code -s', 'ls', 'dir')",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga1_q8_extract_zip_and_read_csv",
                "description": "Extract a zip file and read a value from a CSV file inside it",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the zip file",
                        },
                        "column_name": {
                            "type": "string",
                            "description": "Column name to extract value from",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_zip_and_process_files",
                "description": "Extract a zip file and process multiple files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the zip file",
                        },
                        "operation": {
                            "type": "string",
                            "description": "Operation to perform on files",
                        },
                    },
                    "required": ["file_path", "operation"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga1_q2_api_request",
                "description": "Make an API request to a specified URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to make the request to",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST"],
                            "description": "HTTP method to use",
                        },
                        "headers": {
                            "type": "object",
                            "description": "Headers to include in the request",
                        },
                        "data": {
                            "type": "object",
                            "description": "Data to include in the request body",
                        },
                    },
                    "required": ["url", "method"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga1_q9_sort_json_array",
                "description": "Sort a JSON array based on specified criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "json_array": {
                            "type": "string",
                            "description": "JSON array to sort",
                        },
                        "sort_keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of keys to sort by",
                        },
                    },
                    "required": ["json_array", "sort_keys"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga1_q7_count_days_of_week",
                "description": "Count occurrences of a specific day of the week between two dates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in ISO format (YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in ISO format (YYYY-MM-DD)",
                        },
                        "day_of_week": {
                            "type": "string",
                            "enum": [
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday",
                                "Sunday",
                            ],
                            "description": "Day of the week to count",
                        },
                    },
                    "required": ["start_date", "end_date", "day_of_week"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga1_q12_process_encoded_files",
                "description": "Process files with different encodings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the zip file containing encoded files",
                        },
                        "target_symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of symbols to search for",
                        },
                    },
                    "required": ["file_path", "target_symbols"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga1_q4_q5_calculate_spreadsheet_formula",
                "description": "Calculate the result of a spreadsheet formula",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "formula": {
                            "type": "string",
                            "description": "The formula to calculate",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["google_sheets", "excel"],
                            "description": "Type of spreadsheet",
                        },
                    },
                    "required": ["formula", "type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga1_q17_compare_files",
                "description": "Compare two files and analyze differences",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the zip file containing files to compare",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga1_q18_run_sql_query",
                "description": "Calculate a SQL query result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query to run"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q1_generate_markdown_documentation",
                "description": "Generate markdown documentation with specific elements",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic for the markdown documentation",
                        },
                        "elements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of markdown elements to include",
                        },
                    },
                    "required": ["topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q2_compress_image",
                "description": "Compress an image to a target size while maintaining quality",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the image file",
                        },
                        "target_size": {
                            "type": "integer",
                            "description": "Target size in bytes",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q3_create_github_pages",
                "description": "Generate HTML content for GitHub Pages with email protection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "Email address to include in the page",
                        },
                        "content": {
                            "type": "string",
                            "description": "Optional content for the page",
                        },
                    },
                    "required": ["email"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q4_run_colab_code",
                "description": "Simulate running code on Google Colab",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to run",
                        },
                        "email": {
                            "type": "string",
                            "description": "Email address for authentication",
                        },
                    },
                    "required": ["code", "email"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q5_analyze_image_brightness",
                "description": "Analyze image brightness and count pixels above threshold",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the image file",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Brightness threshold",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q6_deploy_vercel_app",
                "description": "Generate code for a Vercel app deployment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file": {
                            "type": "string",
                            "description": "Path to the data file",
                        },
                        "app_name": {
                            "type": "string",
                            "description": "Optional name for the app",
                        },
                    },
                    "required": ["data_file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q7_create_github_action",
                "description": "Generate GitHub Action workflow with email in step name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "Email to include in step name",
                        },
                        "repository": {
                            "type": "string",
                            "description": "Optional repository name",
                        },
                    },
                    "required": ["email"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q8_create_docker_image",
                "description": "Generate Dockerfile and instructions for Docker Hub deployment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "Tag for the Docker image",
                        },
                        "dockerfile_content": {
                            "type": "string",
                            "description": "Optional Dockerfile content",
                        },
                    },
                    "required": ["tag"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q9_filter_students_by_class",
                "description": "Filter students from a CSV file by class",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file",
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of classes to filter by",
                        },
                    },
                    "required": ["file_path", "classes"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga2_q10_setup_llamafile_with_ngrok",
                "description": "Generate instructions for setting up Llamafile with ngrok",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the Llamafile model",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga3_q1_analyze_sentiment",
                "description": "Analyze sentiment of text using OpenAI API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to analyze for sentiment",
                        },
                        "api_key": {
                            "type": "string",
                            "description": "Optional API key for OpenAI",
                        },
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga3_q2_count_tokens",
                "description": "Count tokens in a message sent to OpenAI API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to count tokens for",
                        },
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga3_q3_generate_structured_output",
                "description": "Generate structured JSON output using OpenAI API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Prompt for generating structured output",
                        },
                        "structure_type": {
                            "type": "string",
                            "description": "Type of structure to generate (e.g., addresses, products)",
                        },
                    },
                    "required": ["prompt", "structure_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga4_q1_count_cricket_ducks",
                "description": "Count the number of ducks in ESPN Cricinfo ODI batting stats for a specific page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_number": {
                            "type": "integer",
                            "description": "Page number to analyze",
                        },
                    },
                    "required": ["page_number"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga4_q2_get_imdb_movies",
                "description": "Get movie information from IMDb with ratings in a specific range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_rating": {
                            "type": "number",
                            "description": "Minimum rating to filter by",
                        },
                        "max_rating": {
                            "type": "number",
                            "description": "Maximum rating to filter by",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of movies to return",
                        },
                    },
                    "required": ["min_rating", "max_rating"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga4_q3_generate_country_outline",
                "description": "Generate a Markdown outline from Wikipedia headings for a country",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "Name of the country",
                        },
                    },
                    "required": ["country"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga4_q4_get_weather_forecast",
                "description": "Get weather forecast for a city using BBC Weather API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Name of the city",
                        },
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga3_q4_generate_vision_api_request",
                "description": "Generate a JSON body for OpenAI's vision API to extract text from an image",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "Base64 URL of the image",
                        },
                    },
                    "required": ["image_url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga3_q5_generate_embeddings_request",
                "description": "Generate a JSON body for OpenAI's embeddings API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of texts to generate embeddings for",
                        },
                    },
                    "required": ["texts"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga3_q6_find_most_similar_phrases",
                "description": "Find the most similar pair of phrases based on cosine similarity of their embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "embeddings_dict": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                            "description": "Dictionary mapping phrases to their embeddings",
                        },
                    },
                    "required": ["embeddings_dict"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_document_similarity",
                "description": "Compute similarity between a query and a list of documents using embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document texts",
                        },
                        "query": {
                            "type": "string",
                            "description": "Query string to compare against documents",
                        },
                    },
                    "required": ["docs", "query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga3_q8_parse_function_call",
                "description": "Parse a natural language query to determine which function to call and extract parameters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_delhi_bounding_box",
                "description": "Get the minimum latitude of Delhi, India using the Nominatim API",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_duckdb_hn_post",
                "description": "Find the latest Hacker News post mentioning DuckDB with at least 71 points",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_newest_seattle_github_user",
                "description": "Find the newest GitHub user in Seattle with over 130 followers",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga4_q8_create_github_action_workflow",
                "description": "Create a GitHub Action workflow that runs daily and adds a commit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "Email to include in the step name",
                        },
                        "repository_url": {
                            "type": "string",
                            "description": "Optional repository URL",
                        },
                    },
                    "required": ["email"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga4_q9_extract_tables_from_pdf",
                "description": "Extract tables from a PDF file and calculate the total Biology marks of students who scored 17 or more marks in Physics in groups 43-66 (inclusive)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the PDF file",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga4_q10_convert_pdf_to_markdown",
                "description": "Convert a PDF file to Markdown and format it with Prettier",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the PDF file",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga5_q1_clean_sales_data_and_calculate_margin",
                "description": "Clean sales data from Excel and calculate margin for filtered transactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the Excel file",
                        },
                        "cutoff_date_str": {
                            "type": "string",
                            "description": "Cutoff date string (e.g., 'Sun Feb 06 2022 18:40:58 GMT+0530 (India Standard Time)')",
                        },
                        "product_filter": {
                            "type": "string",
                            "description": "Product name to filter by (e.g., 'Iota')",
                        },
                        "country_filter": {
                            "type": "string",
                            "description": "Country to filter by after standardization (e.g., 'UK')",
                        },
                    },
                    "required": [
                        "file_path",
                        "cutoff_date_str",
                        "product_filter",
                        "country_filter",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga5_q2_count_unique_students",
                "description": "Count unique students in a text file based on student IDs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the text file with student marks",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga5_q3_analyze_apache_logs",
                "description": "Analyze Apache log files to count requests matching specific criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the Apache log file (can be gzipped)",
                        },
                        "section_path": {
                            "type": "string",
                            "description": "Path section to filter (e.g., '/telugump3/')",
                        },
                        "day_of_week": {
                            "type": "string",
                            "description": "Day to filter (e.g., 'Tuesday')",
                        },
                        "start_hour": {
                            "type": "integer",
                            "description": "Starting hour for time window (inclusive)",
                        },
                        "end_hour": {
                            "type": "integer",
                            "description": "Ending hour for time window (exclusive)",
                        },
                        "request_method": {
                            "type": "string",
                            "description": "HTTP method to filter (e.g., 'GET')",
                        },
                        "status_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Tuple of (min_status, max_status) for HTTP status codes",
                        },
                        "timezone_offset": {
                            "type": "string",
                            "description": "Timezone offset in format '+0000' or '-0500'",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga5_q4_analyze_bandwidth_by_ip",
                "description": "Analyze Apache log files to identify top bandwidth consumers by IP address",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the Apache log file (can be gzipped)",
                        },
                        "section_path": {
                            "type": "string",
                            "description": "Path section to filter (e.g., '/kannada/')",
                        },
                        "specific_date": {
                            "type": "string",
                            "description": "Date to filter in format 'YYYY-MM-DD'",
                        },
                        "timezone_offset": {
                            "type": "string",
                            "description": "Timezone offset in format '+0000' or '-0500'",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_sales_with_phonetic_clustering",
                "description": "Analyze sales data with phonetic clustering for city names",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the sales data file (CSV, Excel, etc.)",
                        },
                        "product_filter": {
                            "type": "string",
                            "description": "Product name to filter by (e.g., 'Soap')",
                        },
                        "min_units": {
                            "type": "integer",
                            "description": "Minimum number of units for filtering",
                        },
                        "target_city": {
                            "type": "string",
                            "description": "Target city to find (will use phonetic matching)",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "parse_partial_json_sales",
                "description": "Parse partial JSON data from a JSONL file and calculate total sales",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the JSONL file with partial JSON data",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga5_q7_count_json_key_occurrences",
                "description": "Count occurrences of a specific key in a nested JSON structure",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the JSON file",
                        },
                        "target_key": {
                            "type": "string",
                            "description": "The key to search for in the JSON structure",
                        },
                    },
                    "required": ["file_path", "target_key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_duckdb_query",
                "description": "Generate and format DuckDB SQL queries for various data analysis tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "description": "Type of query to generate (e.g., 'post_comments', 'user_activity')",
                        },
                        "timestamp_filter": {
                            "type": "string",
                            "description": "ISO timestamp for filtering data (e.g., '2025-02-26T00:17:09.465Z')",
                        },
                        "numeric_filter": {
                            "type": "integer",
                            "description": "Numeric threshold for filtering (e.g., 5 for star count)",
                        },
                        "sort_order": {
                            "type": "string",
                            "description": "Sort order for results ('ASC' or 'DESC')",
                        },
                    },
                    "required": ["query_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "transcribe_youtube_segment",
                "description": "Extract audio from a YouTube video segment and transcribe it",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "youtube_url": {
                            "type": "string",
                            "description": "URL of the YouTube video",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "End time in seconds",
                        },
                    },
                    "required": ["youtube_url", "start_time", "end_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ga5_q10_reconstruct_scrambled_image",
                "description": "Reconstruct an image from scrambled pieces using a mapping",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the scrambled image",
                        },
                        "mapping_data": {
                            "type": "string",
                            "description": "String containing the mapping data (tab or space separated)",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the reconstructed image (optional)",
                        },
                    },
                    "required": ["image_path", "mapping_data"],
                },
            },
        },       
        {
            "type": "function",
            "function": {
                "name": "ga1_q3_calculate_prettier_sha256",
                "description": "Calculate SHA256 hash of a file after formatting with Prettier",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Path to the file to format and hash",
                        }
                    },
                    "required": ["filename"],
                },
            },
        },
    ]