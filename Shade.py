import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SiriusXMAnalyzer:
    def __init__(self):
        """
        Initialise l'analyseur de Sirius XM et sa filiale Shade 45
        """
        self.radios = {
            'Sirius XM Global': {
                'type': 'Satellite Radio',
                'lancement': 2001,
                'pays': 'États-Unis',
                'couleur': '#1F77B4',  # Bleu
                'marqueur': 'o',
                'description': 'Service principal de radio satellite, abonnements mensuels'
            },
            'Shade 45': {
                'type': 'Hip-Hop Channel',
                'lancement': 2004,
                'pays': 'États-Unis',
                'couleur': '#FF7F0E',  # Orange
                'marqueur': 's',
                'description': 'Chaîne hip-hop fondée par Eminem, contenu explicite'
            },
            'Sirius XM Canada': {
                'type': 'Satellite Radio',
                'lancement': 2005,
                'pays': 'Canada',
                'couleur': '#2CA02C',  # Vert
                'marqueur': '^',
                'description': 'Filiale canadienne, adaptation locale'
            },
            'Sirius XM Streaming': {
                'type': 'Digital Service',
                'lancement': 2010,
                'pays': 'États-Unis',
                'couleur': '#D62728',  # Rouge
                'marqueur': 'D',
                'description': 'Service de streaming digital, audience croissante'
            }
        }
        
        # Données historiques simulées (2010-2025) - en millions d'abonnés/auditeurs
        self.historical_data = self._create_detailed_historical_data()
        self.forecasts = {}
        
        # Configuration du style des graphiques
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def _create_detailed_historical_data(self):
        """
        Crée des données historiques détaillées et réalistes pour Sirius XM et ses services
        """
        data = {}
        
        # Sirius XM Global - croissance avec saturation du marché
        global_data = {}
        base_value = 25.1  # Millions d'abonnés en 2010
        for year in range(2010, 2026):
            # Tendance de base avec saturation progressive
            if year <= 2015:
                trend = 2.1  # Forte croissance initiale
            elif year <= 2020:
                trend = 1.3  # Croissance modérée
            else:
                trend = 0.6  # Maturation du marché
            
            # Événements marquants
            if year == 2012:  # Fusion Sirius/XM consolidée
                event_boost = 1.8
            elif year == 2015:  # Expansion des services intégrés véhicules
                event_boost = 2.2
            elif year == 2018:  # Lancement de nouvelles fonctionnalités
                event_boost = 1.5
            elif year == 2020:  # COVID (hausse de l'écoute à domicile)
                event_boost = 2.8
            elif year == 2022:  # Inflation impactant les abonnements
                event_boost = -1.2
            else:
                event_boost = 0
            
            value = base_value + (trend * (year - 2010)) + event_boost
            variation = np.random.normal(0, 0.7)
            value += variation
            
            global_data[year] = max(20.0, min(40.0, round(value, 1)))
        data['Sirius XM Global'] = global_data
        
        # Shade 45 - croissance liée à la popularité du hip-hop
        shade45_data = {}
        base_value = 3.2  # Millions d'auditeurs en 2010
        for year in range(2010, 2026):
            # Tendance basée sur les cycles de popularité du hip-hop
            if year <= 2015:
                trend = 0.25  # Croissance stable
            elif year <= 2020:
                trend = 0.35  # Renaissance du hip-hop
            else:
                trend = 0.2   # Maturation
            
            # Événements spécifiques à Shade 45
            if year == 2013:  # Nouveaux shows exclusifs
                event_boost = 0.8
            elif year == 2017:  # Collaboration avec des artistes majeurs
                event_boost = 0.9
            elif year == 2020:  # Émissions spéciales confinement
                event_boost = 1.2
            elif year == 2023:  # 20ème anniversaire
                event_boost = 0.7
            else:
                event_boost = 0
            
            value = base_value + (trend * (year - 2010)) + event_boost
            variation = np.random.normal(0, 0.3)
            value += variation
            shade45_data[year] = max(2.0, min(8.0, round(value, 1)))
        data['Shade 45'] = shade45_data
        
        # Sirius XM Canada - marché plus petit mais stable
        canada_data = {}
        base_value = 2.8  # Millions d'abonnés en 2010
        for year in range(2010, 2026):
            trend = 0.15  # Croissance lente
            
            if year == 2014:  # Amélioration de l'offre
                event_boost = 0.4
            elif year == 2019:  # Partenariats automobiles
                event_boost = 0.3
            elif year == 2021:  # Effet COVID modéré
                event_boost = 0.5
            else:
                event_boost = 0
            
            value = base_value + (trend * (year - 2010)) + event_boost
            variation = np.random.normal(0, 0.2)
            value += variation
            canada_data[year] = max(2.0, min(5.0, round(value, 1)))
        data['Sirius XM Canada'] = canada_data
        
        # Sirius XM Streaming - croissance exponentielle
        streaming_data = {}
        base_value = 1.5  # Millions d'utilisateurs en 2010
        for year in range(2010, 2026):
            # Croissance accélérée du streaming
            if year <= 2015:
                trend = 0.8
            elif year <= 2020:
                trend = 1.5
            else:
                trend = 2.0
            
            if year == 2016:  # Lancement de l'app mobile améliorée
                event_boost = 1.2
            elif year == 2019:  # Intégration avec assistants vocaux
                event_boost = 1.8
            elif year == 2021:  # Pandémie boostant le streaming
                event_boost = 3.5
            elif year == 2024:  # Nouveaux forfaits famille
                event_boost = 2.0
            else:
                event_boost = 0
            
            value = base_value + (trend * (year - 2010)) + event_boost
            variation = np.random.normal(0, 0.5)
            value += variation
            streaming_data[year] = max(1.0, min(25.0, round(value, 1)))
        data['Sirius XM Streaming'] = streaming_data
        
        return data
    
    def polynomial_regression(self, years, values, degree=2):
        """
        Implémente une régression polynomiale
        """
        X = np.vander(years, degree + 1, increasing=True)
        coefficients = np.linalg.lstsq(X, values, rcond=None)[0]
        
        def predict(x):
            x_poly = np.vander([x], degree + 1, increasing=True)
            return np.dot(x_poly, coefficients).item()
        
        return predict
    
    def predict_radio_audience(self, radio_name, start_year, end_year, degree=2):
        """
        Prédit l'audience pour un service spécifique
        """
        if radio_name not in self.historical_data:
            raise ValueError(f"Données non disponibles pour {radio_name}")
            
        historical = self.historical_data[radio_name]
        years = np.array(list(historical.keys()))
        audiences = np.array(list(historical.values()))
        
        predictor = self.polynomial_regression(years, audiences, degree)
        
        forecast = {}
        for year in range(start_year, end_year + 1):
            prediction = predictor(year)
            
            # Contraintes réalistes selon le service
            if radio_name == "Sirius XM Global":
                min_val, max_val = 20.0, 45.0
            elif radio_name == "Shade 45":
                min_val, max_val = 2.0, 10.0
            elif radio_name == "Sirius XM Canada":
                min_val, max_val = 1.5, 6.0
            else:  # Streaming
                min_val, max_val = 1.0, 35.0
                
            prediction = max(min_val, min(max_val, prediction))
            forecast[year] = round(prediction, 1)
        
        self.forecasts[radio_name] = forecast
        return forecast
    
    def predict_all_radios(self, start_year, end_year, degree=2):
        """
        Prédit l'audience pour tous les services
        """
        all_forecasts = {}
        for radio_name in self.radios.keys():
            forecast = self.predict_radio_audience(radio_name, start_year, end_year, degree)
            all_forecasts[radio_name] = forecast
        
        return all_forecasts
    
    def simulate_scenario(self, scenario_name, base_year, end_year, parameters):
        """
        Simule un scénario pour tous les services
        """
        scenario_results = {}
        
        for radio_name, radio_params in parameters.items():
            if radio_name not in self.historical_data:
                continue
                
            simulated_data = self.historical_data[radio_name].copy()
            current_value = self.historical_data[radio_name][base_year]
            
            for year in range(base_year + 1, end_year + 1):
                current_value += radio_params.get('base_trend', 0)
                
                if 'volatility' in radio_params:
                    variation = np.random.normal(0, radio_params['volatility'])
                    current_value += variation
                
                for event_year, event_impact in radio_params.get('events', {}).items():
                    if year == event_year:
                        current_value += event_impact
                
                if radio_name == "Sirius XM Global":
                    min_val, max_val = 20.0, 45.0
                elif radio_name == "Shade 45":
                    min_val, max_val = 2.0, 10.0
                elif radio_name == "Sirius XM Canada":
                    min_val, max_val = 1.5, 6.0
                else:  # Streaming
                    min_val, max_val = 1.0, 35.0
                    
                current_value = max(min_val, min(max_val, current_value))
                simulated_data[year] = round(current_value, 1)
            
            scenario_results[radio_name] = simulated_data
        
        return scenario_results
    
    def calculate_kpis(self):
        """
        Calcule les indicateurs de performance clés
        """
        kpis = {}
        
        for radio_name in self.radios.keys():
            historical = self.historical_data[radio_name]
            years = list(historical.keys())
            values = list(historical.values())
            
            mean_audience = np.mean(values)
            std_audience = np.std(values)
            min_audience = min(values)
            max_audience = max(values)
            min_year = years[values.index(min_audience)]
            max_year = years[values.index(max_audience)]
            
            recent_years = [year for year in years if year >= 2020]
            recent_values = [historical[year] for year in recent_years]
            if len(recent_values) > 1:
                trend_5y = recent_values[-1] - recent_values[0]
            else:
                trend_5y = 0
            
            volatility = (std_audience / mean_audience) * 100 if mean_audience > 0 else 0
            
            kpis[radio_name] = {
                'audience_moyenne': round(mean_audience, 2),
                'ecart_type': round(std_audience, 2),
                'audience_min': min_audience,
                'annee_min': min_year,
                'audience_max': max_audience,
                'annee_max': max_year,
                'tendance_5ans': round(trend_5y, 2),
                'volatilite': round(volatility, 2)
            }
        
        return kpis
    
    def create_detailed_analysis(self, end_year=2030):
        """
        Crée une analyse détaillée avec multiples visualisations
        """
        forecasts = self.predict_all_radios(2026, end_year)
        
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(4, 3)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])
        ax5 = fig.add_subplot(gs[2, :])
        ax6 = fig.add_subplot(gs[3, 0])
        ax7 = fig.add_subplot(gs[3, 1])
        ax8 = fig.add_subplot(gs[3, 2])
        
        # 1. Évolution historique et prévisionnelle
        for radio_name in self.radios.keys():
            historical = self.historical_data[radio_name]
            forecast = forecasts[radio_name]
            
            all_years = list(historical.keys()) + list(forecast.keys())
            all_values = list(historical.values()) + list(forecast.values())
            
            split_index = len(historical)
            historical_years = all_years[:split_index]
            historical_values = all_values[:split_index]
            forecast_years = all_years[split_index:]
            forecast_values = all_values[split_index:]
            
            color = self.radios[radio_name]['couleur']
            marker = self.radios[radio_name]['marqueur']
            
            ax1.plot(historical_years, historical_values, marker=marker, color=color, 
                    linewidth=2, markersize=6, label=radio_name)
            ax1.plot(forecast_years, forecast_values, '--', color=color, linewidth=2, alpha=0.7)
        
        ax1.axvline(x=2025, color='gray', linestyle='--', alpha=0.7)
        ax1.text(2025.2, ax1.get_ylim()[1] * 0.9, 'Prévisions', fontsize=10)
        ax1.set_title('Évolution des Audiences de Sirius XM et ses Services (2010-2030)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Année')
        ax1.set_ylabel('Audience (millions)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Comparaison des audiences en 2030
        latest_year = end_year
        latest_audiences = {name: data[latest_year] for name, data in forecasts.items()}
        
        names = list(latest_audiences.keys())
        values = list(latest_audiences.values())
        colors = [self.radios[name]['couleur'] for name in names]
        
        bars = ax2.bar(names, values, color=colors)
        ax2.set_title(f'Audience Prévue en {latest_year}', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Audience (millions)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}M', ha='center', va='bottom')
        
        # 3. Répartition par type de service en 2030
        service_types = {}
        for radio_name, audience in latest_audiences.items():
            service_type = self.radios[radio_name]['type']
            if service_type in service_types:
                service_types[service_type] += audience
            else:
                service_types[service_type] = audience
        
        wedges, texts, autotexts = ax3.pie(service_types.values(), labels=service_types.keys(), autopct='%1.1f%%',
               startangle=90, colors=['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728'])
        ax3.set_title(f'Répartition par Type de Service en {latest_year}', fontsize=14, fontweight='bold')
        
        # 4. Croissance prévue (2025-2030)
        growth_data = {}
        for radio_name in self.radios.keys():
            growth = forecasts[radio_name][2030] - self.historical_data[radio_name][2025]
            growth_data[radio_name] = growth
        
        names = list(growth_data.keys())
        values = list(growth_data.values())
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        bars = ax4.bar(names, values, color=colors)
        ax4.set_title('Croissance Prévisionnelle (2025-2030)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Variation (millions)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            y_offset = 0.1 if height >= 0 else -0.1
            ax4.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{height:+.1f}M', ha='center', va=va)
        
        # 5. Heatmap des audiences
        all_years = sorted(set().union(*[d.keys() for d in self.historical_data.values()]))
        heatmap_data = []
        
        for radio_name in self.radios.keys():
            row = []
            for year in all_years:
                if year in self.historical_data[radio_name]:
                    row.append(self.historical_data[radio_name][year])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        im = ax5.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax5.set_xticks(range(len(all_years)))
        ax5.set_xticklabels(all_years, rotation=45)
        ax5.set_yticks(range(len(self.radios)))
        ax5.set_yticklabels(list(self.radios.keys()))
        ax5.set_title('Heatmap des Audiences par Année et par Service', fontsize=14, fontweight='bold')
        
        for i in range(len(self.radios)):
            for j in range(len(all_years)):
                if not np.isnan(heatmap_data[i][j]):
                    text = ax5.text(j, i, f'{heatmap_data[i][j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        fig.colorbar(im, ax=ax5, label='Audience (millions)')
        
        # 6. Volatilité historique
        kpis = self.calculate_kpis()
        volatilities = [kpis[name]['volatilite'] for name in self.radios.keys()]
        
        bars = ax6.bar(list(self.radios.keys()), volatilities, 
                      color=[self.radios[name]['couleur'] for name in self.radios.keys()])
        ax6.set_title('Volatilité Historique des Audiences', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Volatilité (%)')
        ax6.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 7. Parts de marché en 2030
        total_2030 = sum(forecasts[radio_name][2030] for radio_name in self.radios.keys())
        market_shares = {name: (forecasts[name][2030] / total_2030) * 100 for name in self.radios.keys()}
        
        wedges, texts, autotexts = ax7.pie(market_shares.values(), labels=market_shares.keys(), autopct='%1.1f%%',
               startangle=90, colors=[self.radios[name]['couleur'] for name in market_shares.keys()])
        ax7.set_title(f'Parts de Marché en {latest_year}', fontsize=14, fontweight='bold')
        
        # 8. Performance relative
        normalized_data = {}
        for radio_name in self.radios.keys():
            max_audience = max(self.historical_data[radio_name].values())
            normalized_data[radio_name] = [val / max_audience * 100 for val in self.historical_data[radio_name].values()]
        
        years = list(self.historical_data['Sirius XM Global'].keys())
        for radio_name in self.radios.keys():
            ax8.plot(years, normalized_data[radio_name], 
                    color=self.radios[radio_name]['couleur'],
                    marker=self.radios[radio_name]['marqueur'],
                    linewidth=2, markersize=4, label=radio_name)
        
        ax8.set_title('Performance Relative (Normalisée)', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Année')
        ax8.set_ylabel('Performance (% du maximum historique)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analyse_detaillée_sirius_xm.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.generate_comprehensive_report(forecasts)
        self.create_additional_dashboards(forecasts)
    
    def generate_comprehensive_report(self, forecasts):
        """
        Génère un rapport détaillé des prévisions
        """
        print("=" * 80)
        print("RAPPORT COMPLET D'ANALYSE DE SIRIUS XM")
        print("=" * 80)
        
        kpis = self.calculate_kpis()
        
        for radio_name in self.radios.keys():
            historical = self.historical_data[radio_name]
            forecast = forecasts[radio_name]
            
            hist_avg = np.mean(list(historical.values()))
            fcst_avg = np.mean(list(forecast.values()))
            growth_5y = forecast[2030] - historical[2025]
            growth_percent = (growth_5y / historical[2025]) * 100
            
            recent_trend = "🟢 Hausse" if growth_5y > 0 else "🔴 Baisse" if growth_5y < 0 else "🟡 Stable"
            
            print(f"\n📻 {radio_name} ({self.radios[radio_name]['pays']})")
            print(f"   {self.radios[radio_name]['description']}")
            print(f"   📊 Audience moyenne historique: {hist_avg:.1f}M")
            print(f"   📈 Audience moyenne prévisionnelle: {fcst_avg:.1f}M")
            print(f"   🎯 Audience prévue en 2030: {forecast[2030]:.1f}M")
            print(f"   📈 Croissance 2025-2030: {growth_5y:+.1f}M ({growth_percent:+.1f}%) {recent_trend}")
            print(f"   📉 Audience minimale: {kpis[radio_name]['audience_min']}M ({kpis[radio_name]['annee_min']})")
            print(f"   📈 Audience maximale: {kpis[radio_name]['audience_max']}M ({kpis[radio_name]['annee_max']})")
            print(f"   📊 Volatilité: {kpis[radio_name]['volatilite']}%")
        
        print("\n" + "=" * 80)
        print("ANALYSE COMPARATIVE AVANCÉE")
        print("=" * 80)
        
        growth_rates = {}
        for radio_name in self.radios.keys():
            growth = forecasts[radio_name][2030] - self.historical_data[radio_name][2025]
            growth_rates[radio_name] = growth
        
        fastest_growing = max(growth_rates, key=growth_rates.get)
        fastest_declining = min(growth_rates, key=growth_rates.get)
        
        print(f"   🚀 Croissance la plus forte: {fastest_growing} ({growth_rates[fastest_growing]:+.1f}M)")
        print(f"   📉 Croissance la plus faible: {fastest_declining} ({growth_rates[fastest_declining]:+.1f}M)")
        
        total_2030 = sum(forecasts[radio_name][2030] for radio_name in self.radios.keys())
        market_shares = {name: (forecasts[name][2030] / total_2030) * 100 for name in self.radios.keys()}
        
        print(f"\n   📊 Parts de marché prévues en 2030:")
        for radio_name, share in market_shares.items():
            print(f"      {radio_name}: {share:.1f}%")
        
        total_audience_2025 = sum(self.historical_data[radio_name][2025] for radio_name in self.radios.keys())
        total_audience_2030 = sum(forecasts[radio_name][2030] for radio_name in self.radios.keys())
        network_growth = total_audience_2030 - total_audience_2025
        network_growth_percent = (network_growth / total_audience_2025) * 100
        
        print(f"\n   🌐 Performance globale du réseau:")
        print(f"      Audience totale 2025: {total_audience_2025:.1f}M")
        print(f"      Audience totale 2030: {total_audience_2030:.1f}M")
        print(f"      Croissance du réseau: {network_growth:+.1f}M ({network_growth_percent:+.1f}%)")
        
        print(f"\n" + "=" * 80)
        print("RECOMMANDATIONS STRATÉGIQUES")
        print("=" * 80)
        
        if growth_rates['Sirius XM Streaming'] > 3.0:
            print("   🔸 Sirius XM Streaming montre une forte croissance. Recommandations:")
            print("      - Investir dans l'infrastructure streaming")
            print("      - Développer des contenus exclusifs digitaux")
            print("      - Renforcer les partenariats avec fabricants d'appareils")
        
        if growth_rates['Shade 45'] > 0.5:
            print("   🔸 Shade 45 maintient une croissance positive. Recommandations:")
            print("      - Capitaliser sur la marque Eminem et le hip-hop")
            print("      - Développer des contenus vidéo complémentaires")
            print("      - Élargir l'offre vers les marchés internationaux")
        
        print(f"\n   🔮 Scénarios alternatifs à considérer:")
        print("      - Scénario technologique: Intégration avancée IA et personnalisation")
        print("      - Scénario contenu: Développement de productions originales exclusives")
        print("      - Scénario marché: Expansion internationale ciblée")
    
    def create_additional_dashboards(self, forecasts):
        """
        Crée des dashboards supplémentaires
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Tendance à court terme (2020-2030)
        for radio_name in self.radios.keys():
            years = [y for y in range(2020, 2031)]
            values = []
            for year in years:
                if year <= 2025:
                    values.append(self.historical_data[radio_name][year])
                else:
                    values.append(forecasts[radio_name][year])
            
            ax1.plot(years, values, color=self.radios[radio_name]['couleur'],
                    marker=self.radios[radio_name]['marqueur'], linewidth=2, label=radio_name)
        
        ax1.set_title('Tendance à Court Terme (2020-2030)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Année')
        ax1.set_ylabel('Audience (millions)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=2025, color='gray', linestyle='--', alpha=0.7)
        
        # Analyse de la volatilité
        kpis = self.calculate_kpis()
        radio_names = list(self.radios.keys())
        volatilities = [kpis[name]['volatilite'] for name in radio_names]
        avg_audiences = [kpis[name]['audience_moyenne'] for name in radio_names]
        colors = [self.radios[name]['couleur'] for name in radio_names]
        
        scatter = ax2.scatter(avg_audiences, volatilities, s=100, c=colors, alpha=0.7)
        ax2.set_title('Relation entre Audience Moyenne et Volatilité', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Audience Moyenne (millions)')
        ax2.set_ylabel('Volatilité (%)')
        ax2.grid(True, alpha=0.3)
        
        for i, name in enumerate(radio_names):
            ax2.annotate(name, (avg_audiences[i], volatilities[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('dashboard_complementaire_sirius_xm.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_to_excel(self, filename="sirius_xm_analysis_detailed.xlsx"):
        """
        Exporte toutes les données vers un fichier Excel
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Données historiques
            hist_data = []
            for radio_name, data in self.historical_data.items():
                for year, audience in data.items():
                    hist_data.append({
                        'Service': radio_name,
                        'Année': year,
                        'Audience': audience,
                        'Type': 'Historique',
                        'Pays': self.radios[radio_name]['pays'],
                        'Type Service': self.radios[radio_name]['type'],
                        'Année Lancement': self.radios[radio_name]['lancement']
                    })
            
            hist_df = pd.DataFrame(hist_data)
            hist_df.to_excel(writer, sheet_name='Données Historiques', index=False)
            
            # Prévisions
            if not self.forecasts:
                self.predict_all_radios(2026, 2030)
                
            forecast_data = []
            for radio_name, data in self.forecasts.items():
                for year, audience in data.items():
                    forecast_data.append({
                        'Service': radio_name,
                        'Année': year,
                        'Audience': audience,
                        'Type': 'Prévision',
                        'Pays': self.radios[radio_name]['pays'],
                        'Type Service': self.radios[radio_name]['type'],
                        'Année Lancement': self.radios[radio_name]['lancement']
                    })
            
            forecast_df = pd.DataFrame(forecast_data)
            forecast_df.to_excel(writer, sheet_name='Prévisions', index=False)
            
            # Données combinées
            combined_df = pd.concat([hist_df, forecast_df])
            combined_df.to_excel(writer, sheet_name='Toutes Données', index=False)
            
            # Résumé avec KPIs
            kpis = self.calculate_kpis()
            summary_data = []
            for radio_name in self.radios.keys():
                hist_avg = np.mean(list(self.historical_data[radio_name].values()))
                fcst_avg = np.mean(list(self.forecasts[radio_name].values()))
                growth = self.forecasts[radio_name][2030] - self.historical_data[radio_name][2025]
                
                summary_data.append({
                    'Service': radio_name,
                    'Pays': self.radios[radio_name]['pays'],
                    'Audience Moyenne Historique': hist_avg,
                    'Audience Moyenne Prévisionnelle': fcst_avg,
                    'Croissance 2025-2030': growth,
                    'Audience 2030': self.forecasts[radio_name][2030],
                    'Volatilité (%)': kpis[radio_name]['volatilite'],
                    'Audience Min': kpis[radio_name]['audience_min'],
                    'Année Min': kpis[radio_name]['annee_min'],
                    'Audience Max': kpis[radio_name]['audience_max'],
                    'Année Max': kpis[radio_name]['annee_max']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Résumé', index=False)
        
        print(f"\n✅ Données exportées vers {filename}")


# Exécution du programme
if __name__ == "__main__":
    print("🔍 Analyse de Sirius XM et de ses services...")
    
    # Création de l'analyseur
    analyzer = SiriusXMAnalyzer()
    
    # Génération des prévisions
    print("📊 Génération des prévisions...")
    forecasts = analyzer.predict_all_radios(2026, 2030)
    
    # Création de l'analyse détaillée
    print("📈 Création des visualisations...")
    analyzer.create_detailed_analysis(2030)
    
    # Simulation de scénarios alternatifs
    print("🔮 Simulation de scénarios...")
    scenarios = {
        'Sirius XM Global': {
            'base_trend': 0.3,
            'volatility': 0.4,
            'events': {2027: 1.0, 2029: 1.2}
        },
        'Shade 45': {
            'base_trend': 0.25,
            'volatility': 0.2,
            'events': {2026: 0.5, 2028: 0.7}
        },
        'Sirius XM Canada': {
            'base_trend': 0.1,
            'volatility': 0.15,
            'events': {2027: 0.3}
        },
        'Sirius XM Streaming': {
            'base_trend': 1.8,
            'volatility': 0.6,
            'events': {2026: 2.0, 2028: 2.5}
        }
    }
    
    scenario_results = analyzer.simulate_scenario("Scénario optimiste", 2025, 2030, scenarios)
    
    # Export des données
    print("💾 Export des données...")
    analyzer.export_to_excel()
    
    print("\n🎯 Analyse terminée avec succès!")
    print("📋 Consultez les graphiques générés et le fichier Excel pour les résultats détaillés.")