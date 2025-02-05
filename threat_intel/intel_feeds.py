import requests
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import pandas as pd

class ThreatIntel:
    def __init__(self):
        # Initialize API keys from environment variables
        self.abuseipdb_key = os.getenv('ABUSEIPDB_API_KEY', '')
        self.virustotal_key = os.getenv('VIRUSTOTAL_API_KEY', '')
        
        # Cache for IP lookups to avoid rate limiting
        self.ip_cache = {}
        self.cache_duration = timedelta(hours=1)

    def check_ip_abuseipdb(self, ip: str) -> Dict:
        """Check IP against AbuseIPDB"""
        # Check cache first
        if ip in self.ip_cache:
            if datetime.now() - self.ip_cache[ip]['timestamp'] < self.cache_duration:
                return self.ip_cache[ip]['data']

        url = 'https://api.abuseipdb.com/api/v2/check'
        headers = {
            'Accept': 'application/json',
            'Key': self.abuseipdb_key
        }
        params = {
            'ipAddress': ip,
            'maxAgeInDays': 90
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()['data']
                # Cache the result
                self.ip_cache[ip] = {
                    'timestamp': datetime.now(),
                    'data': data
                }
                return data
            return None
        except Exception as e:
            print(f"Error checking AbuseIPDB: {str(e)}")
            return None

    def check_ip_virustotal(self, ip: str) -> Dict:
        """Check IP against VirusTotal"""
        if not self.virustotal_key:
            return None

        url = f'https://www.virustotal.com/api/v3/ip_addresses/{ip}'
        headers = {
            'x-apikey': self.virustotal_key
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()['data']
            return None
        except Exception as e:
            print(f"Error checking VirusTotal: {str(e)}")
            return None

    def enrich_traffic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich network traffic data with threat intelligence"""
        # Create new columns for threat data
        df['threat_score'] = 0
        df['threat_categories'] = ''
        df['last_reported'] = ''
        
        # Process each unique IP
        # Convert numpy arrays to pandas Series before concatenation
        src_ips = pd.Series(df['Src IP'].unique())
        dst_ips = pd.Series(df['Dst IP'].unique())
        unique_ips = pd.concat([src_ips, dst_ips]).unique()

        for ip in unique_ips:
            # Check AbuseIPDB
            abuse_data = self.check_ip_abuseipdb(ip)
            if abuse_data:
                # Update rows where IP appears as source
                mask_src = df['Src IP'] == ip
                df.loc[mask_src, 'threat_score'] = abuse_data.get('abuseConfidenceScore', 0)
                df.loc[mask_src, 'threat_categories'] = ','.join(
                    str(x) for x in abuse_data.get('reports', [])
                )
                df.loc[mask_src, 'last_reported'] = abuse_data.get('lastReportedAt', '')

                # Update rows where IP appears as destination
                mask_dst = df['Dst IP'] == ip
                df.loc[mask_dst, 'threat_score'] = abuse_data.get('abuseConfidenceScore', 0)
                df.loc[mask_dst, 'threat_categories'] = ','.join(
                    str(x) for x in abuse_data.get('reports', [])
                )
                df.loc[mask_dst, 'last_reported'] = abuse_data.get('lastReportedAt', '')

        return df

    def get_threat_summary(self, ip: str) -> Dict:
        """Get comprehensive threat summary for an IP"""
        summary = {
            'ip': ip,
            'threat_level': 'Unknown',
            'confidence_score': 0,
            'recent_reports': [],
            'categories': [],
            'last_reported': None,
            'country': None,
            'isp': None
        }

        # Check AbuseIPDB
        abuse_data = self.check_ip_abuseipdb(ip)
        if abuse_data:
            summary.update({
                'confidence_score': abuse_data.get('abuseConfidenceScore', 0),
                'country': abuse_data.get('countryCode'),
                'isp': abuse_data.get('isp'),
                'last_reported': abuse_data.get('lastReportedAt')
            })

        # Check VirusTotal
        vt_data = self.check_ip_virustotal(ip)
        if vt_data:
            summary['vt_stats'] = vt_data.get('attributes', {}).get('last_analysis_stats', {})

        # Determine threat level
        if summary['confidence_score'] >= 80:
            summary['threat_level'] = 'High'
        elif summary['confidence_score'] >= 50:
            summary['threat_level'] = 'Medium'
        elif summary['confidence_score'] > 0:
            summary['threat_level'] = 'Low'
        else:
            summary['threat_level'] = 'None'

        return summary 