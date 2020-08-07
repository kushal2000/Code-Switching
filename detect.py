import os, requests, uuid, json

key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
if not key_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(key_var_name))
subscription_key = os.environ[key_var_name]

endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
if not endpoint_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
endpoint = os.environ[endpoint_var_name]

path = '/detect?api-version=3.0'
constructed_url = endpoint + path
print(constructed_url)
headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4()),
    'Ocp-Apim-Subscription-Region': 'eastasia'
}

def detect(word):
    body = [{
        'text': word
    }]

    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    # print(json.dumps(response, sort_keys=True, indent=4,
    #                 ensure_ascii=False, separators=(',', ': ')))
    
    return response[0]['language']

if __name__=='__main__':
    print(detect('Hello'))