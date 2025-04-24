class PrivacyConfig:
    _api_key = "xxxxxxxxxx"
    _organization = "xxxxxxxxxx"
    _project = "xxxxxxxxxx"

    @property
    def gpt_api_key(self):
        return self._api_key
    
    @property
    def organization(self):
        return self._organization
    
    @property
    def project(self):
        return self._project
