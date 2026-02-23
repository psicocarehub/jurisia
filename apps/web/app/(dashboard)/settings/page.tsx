'use client';

import { useState, useEffect } from 'react';
import { Settings, User, Bell, Shield, Building2 } from 'lucide-react';

interface UserProfile {
  name: string;
  email: string;
  oab: string;
}

interface Preferences {
  defaultModel: string;
  notifications: boolean;
  alertAreas: string[];
  alertTribunais: string[];
}

const LLM_MODELS = [
  { value: 'auto', label: 'Automático (Router Inteligente)' },
  { value: 'gaia', label: 'GAIA 4B (Local)' },
  { value: 'deepseek', label: 'DeepSeek V3.2' },
  { value: 'qwen', label: 'Qwen 3.5' },
  { value: 'gemini', label: 'Gemini 3 Pro' },
  { value: 'claude', label: 'Claude Opus 4.6' },
];

const AREAS = [
  'Cível', 'Trabalhista', 'Criminal', 'Tributário', 'Família',
  'Consumidor', 'Administrativo', 'Previdenciário',
];

const TRIBUNAIS = [
  'STF', 'STJ', 'TST', 'TJSP', 'TJRJ', 'TJMG', 'TJRS', 'TJPR',
];

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<'profile' | 'preferences' | 'alerts'>('profile');
  const [profile, setProfile] = useState<UserProfile>({
    name: '',
    email: '',
    oab: '',
  });
  const [prefs, setPrefs] = useState<Preferences>({
    defaultModel: 'auto',
    notifications: true,
    alertAreas: [],
    alertTribunais: [],
  });
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('jurisai_prefs');
    if (stored) {
      try {
        setPrefs(JSON.parse(stored));
      } catch {
        // ignore
      }
    }
    const storedProfile = localStorage.getItem('jurisai_profile');
    if (storedProfile) {
      try {
        setProfile(JSON.parse(storedProfile));
      } catch {
        // ignore
      }
    }
  }, []);

  const handleSave = () => {
    localStorage.setItem('jurisai_prefs', JSON.stringify(prefs));
    localStorage.setItem('jurisai_profile', JSON.stringify(profile));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const toggleArea = (area: string) => {
    setPrefs((p) => ({
      ...p,
      alertAreas: p.alertAreas.includes(area)
        ? p.alertAreas.filter((a) => a !== area)
        : [...p.alertAreas, area],
    }));
  };

  const toggleTribunal = (t: string) => {
    setPrefs((p) => ({
      ...p,
      alertTribunais: p.alertTribunais.includes(t)
        ? p.alertTribunais.filter((x) => x !== t)
        : [...p.alertTribunais, t],
    }));
  };

  const tabs = [
    { id: 'profile' as const, label: 'Perfil', icon: User },
    { id: 'preferences' as const, label: 'Preferências', icon: Settings },
    { id: 'alerts' as const, label: 'Inscrições de Alertas', icon: Bell },
  ];

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="flex items-center gap-2 mb-6">
        <Settings className="h-6 w-6 text-legal-blue-600" />
        <h1 className="text-xl font-semibold text-gray-900">Configurações</h1>
      </div>

      <div className="flex gap-2 mb-6">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`px-4 py-2 text-sm rounded-lg font-medium transition-colors flex items-center gap-1.5 ${
              activeTab === id
                ? 'bg-legal-blue-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>

      <div className="bg-white rounded-xl border p-6">
        {activeTab === 'profile' && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-700 mb-4">Informações Pessoais</h3>
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Nome completo</label>
              <input className="input-field text-sm" value={profile.name} onChange={(e) => setProfile({ ...profile, name: e.target.value })} />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">E-mail</label>
              <input type="email" className="input-field text-sm" value={profile.email} onChange={(e) => setProfile({ ...profile, email: e.target.value })} />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Registro OAB</label>
              <input className="input-field text-sm" placeholder="Ex: SP 123456" value={profile.oab} onChange={(e) => setProfile({ ...profile, oab: e.target.value })} />
            </div>
          </div>
        )}

        {activeTab === 'preferences' && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-700 mb-4">Preferências do Sistema</h3>
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Modelo LLM padrão</label>
              <select className="input-field text-sm" value={prefs.defaultModel} onChange={(e) => setPrefs({ ...prefs, defaultModel: e.target.value })}>
                {LLM_MODELS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="notifications"
                checked={prefs.notifications}
                onChange={(e) => setPrefs({ ...prefs, notifications: e.target.checked })}
                className="h-4 w-4 rounded border-gray-300 text-legal-blue-600"
              />
              <label htmlFor="notifications" className="text-sm text-gray-700">
                Receber notificações de alertas legislativos
              </label>
            </div>
          </div>
        )}

        {activeTab === 'alerts' && (
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-3">Áreas do Direito</h3>
              <p className="text-xs text-gray-500 mb-3">Selecione as áreas para receber alertas de mudanças legislativas</p>
              <div className="flex flex-wrap gap-2">
                {AREAS.map((area) => (
                  <button
                    key={area}
                    onClick={() => toggleArea(area)}
                    className={`px-3 py-1.5 text-sm rounded-full border transition-colors ${
                      prefs.alertAreas.includes(area)
                        ? 'bg-legal-blue-600 text-white border-legal-blue-600'
                        : 'bg-white text-gray-600 border-gray-300 hover:border-legal-blue-400'
                    }`}
                  >
                    {area}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-3">Tribunais</h3>
              <p className="text-xs text-gray-500 mb-3">Selecione os tribunais para monitorar</p>
              <div className="flex flex-wrap gap-2">
                {TRIBUNAIS.map((t) => (
                  <button
                    key={t}
                    onClick={() => toggleTribunal(t)}
                    className={`px-3 py-1.5 text-sm rounded-full border transition-colors ${
                      prefs.alertTribunais.includes(t)
                        ? 'bg-legal-blue-600 text-white border-legal-blue-600'
                        : 'bg-white text-gray-600 border-gray-300 hover:border-legal-blue-400'
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="flex justify-end mt-6 pt-4 border-t">
          {saved && <span className="text-sm text-green-600 mr-3 mt-2">Salvo com sucesso!</span>}
          <button onClick={handleSave} className="btn-primary text-sm">
            Salvar Alterações
          </button>
        </div>
      </div>
    </div>
  );
}
