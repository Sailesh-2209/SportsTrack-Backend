# SportsTrack Backend

* Start the server by using
	```
	python main.py
	```

## API Documentation

### Methods

#### Authentication Methods

1. `/auth/signin`	

	* Accepted methods  
		POST
	
	* Request body
		```json
		{
			"email": "example@email.com",
			"password": "password"
		}
		```
	* Response object
		1. If credentials are valid  
			`RESPONSE_CODE: 200`
			```json
			{
				"token": "token"
			}
			```
		2. If there is an error
			```json
			{
				"error_code": "0/1",
				"message": "message"
			}
			```
			* Error code `0` indicates email error
			* Error code `1` indicates password error
