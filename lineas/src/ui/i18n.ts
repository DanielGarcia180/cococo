/** Textos de la interfaz. Español por defecto; inglés como segundo idioma. */
type Dict = Record<string, string>;

const es: Dict = {
  'app.title': 'LÍNEAS',
  'hud.cash': 'Caja', 'hud.value': 'Valor', 'hud.rep': 'Reputación', 'hud.pause': 'Pausa', 'hud.speed': 'Velocidad',
  'menu.finance': 'Finanzas', 'menu.data': 'Datos', 'menu.research': 'Investigación', 'menu.vehicles': 'Vehículos', 'menu.fund': 'Fondo',
  'menu.challenges': 'Retos', 'menu.prestige': 'Prestigio', 'menu.settings': 'Ajustes', 'menu.photo': 'Modo foto', 'menu.more': 'Más', 'menu.main': 'Menú', 'menu.mods': 'Mods', 'menu.collection': 'Colección',
  'tool.select': 'Seleccionar', 'tool.station': 'Estación', 'tool.track': 'Vía', 'tool.line': 'Línea', 'tool.waypoint': 'Punto de paso', 'tool.bulldoze': 'Demoler', 'tool.heat': 'Mapa de demanda', 'tool.undo': 'Deshacer', 'tool.redo': 'Rehacer',
  'kind.road': 'Carretera', 'kind.rail': 'Ferrocarril', 'kind.metro': 'Metro', 'kind.hsr': 'Alta velocidad',
  'mode.bus': 'Autobús', 'mode.tram': 'Tranvía', 'mode.train': 'Tren', 'mode.metro': 'Metro', 'mode.hsr': 'Alta velocidad', 'mode.truck': 'Camión', 'mode.freight': 'Tren de mercancías',
  'cargo.ore': 'Mineral', 'cargo.steel': 'Acero', 'cargo.goods': 'Bienes', 'cargo.wood': 'Madera', 'cargo.planks': 'Tablones', 'cargo.food': 'Alimentos', 'cargo.oil': 'Petróleo', 'cargo.fuel': 'Combustible',
  'weather.clear': 'Despejado', 'weather.cloud': 'Nublado', 'weather.rain': 'Lluvia', 'weather.fog': 'Niebla', 'weather.snow': 'Nieve',
  'common.close': 'Cerrar', 'common.buy': 'Comprar', 'common.sell': 'Vender', 'common.delete': 'Eliminar', 'common.cancel': 'Cancelar', 'common.confirm': 'Confirmar', 'common.rush': 'Acelerar', 'common.name': 'Nombre', 'common.cost': 'Coste', 'common.days': 'días', 'common.month': 'mes', 'common.export': 'Exportar CSV',
  'tut.0': 'Elige la herramienta Estación (tecla 2) y coloca una parada dentro de la ciudad grande.',
  'tut.1': 'Coloca una segunda estación a 1–3 km de la primera, cubriendo otros barrios.',
  'tut.2': 'Con la herramienta Vía (3), haz clic en una estación y luego en la otra para trazar el ferrocarril.',
  'tut.3': 'Crea una línea (4): haz clic en las dos estaciones en orden.',
  'tut.4': 'Compra un tranvía para la línea desde su panel. Acelera las obras si no quieres esperar.',
  'tut.5': '¡Primer ingreso! Mira el panel de la línea: factor de carga, beneficio y previsión. Ahora amplía la red.',
};
const en: Dict = {
  'app.title': 'LINES',
  'hud.cash': 'Cash', 'hud.value': 'Value', 'hud.rep': 'Reputation', 'hud.pause': 'Pause', 'hud.speed': 'Speed',
  'menu.finance': 'Finance', 'menu.data': 'Data', 'menu.research': 'Research', 'menu.vehicles': 'Vehicles', 'menu.fund': 'Fund',
  'menu.challenges': 'Challenges', 'menu.prestige': 'Prestige', 'menu.settings': 'Settings', 'menu.photo': 'Photo mode', 'menu.more': 'More', 'menu.main': 'Menu', 'menu.mods': 'Mods', 'menu.collection': 'Collection',
  'tool.select': 'Select', 'tool.station': 'Station', 'tool.track': 'Track', 'tool.line': 'Line', 'tool.waypoint': 'Waypoint', 'tool.bulldoze': 'Demolish', 'tool.heat': 'Demand map', 'tool.undo': 'Undo', 'tool.redo': 'Redo',
  'kind.road': 'Road', 'kind.rail': 'Rail', 'kind.metro': 'Metro', 'kind.hsr': 'High speed',
  'mode.bus': 'Bus', 'mode.tram': 'Tram', 'mode.train': 'Train', 'mode.metro': 'Metro', 'mode.hsr': 'High speed', 'mode.truck': 'Truck', 'mode.freight': 'Freight train',
  'cargo.ore': 'Ore', 'cargo.steel': 'Steel', 'cargo.goods': 'Goods', 'cargo.wood': 'Wood', 'cargo.planks': 'Planks', 'cargo.food': 'Food', 'cargo.oil': 'Oil', 'cargo.fuel': 'Fuel',
  'weather.clear': 'Clear', 'weather.cloud': 'Cloudy', 'weather.rain': 'Rain', 'weather.fog': 'Fog', 'weather.snow': 'Snow',
  'common.close': 'Close', 'common.buy': 'Buy', 'common.sell': 'Sell', 'common.delete': 'Delete', 'common.cancel': 'Cancel', 'common.confirm': 'Confirm', 'common.rush': 'Rush', 'common.name': 'Name', 'common.cost': 'Cost', 'common.days': 'days', 'common.month': 'month', 'common.export': 'Export CSV',
  'tut.0': 'Pick the Station tool (key 2) and place a stop inside the big city.',
  'tut.1': 'Place a second station 1–3 km away, covering other districts.',
  'tut.2': 'With the Track tool (3), click one station and then the other to lay rail.',
  'tut.3': 'Create a line (4): click both stations in order.',
  'tut.4': 'Buy a tram for the line from its panel. Rush construction if you do not want to wait.',
  'tut.5': 'First revenue! Check the line panel: load factor, profit and forecast. Now grow the network.',
};

let lang: 'es' | 'en' = 'es';
export function setLang(l: 'es' | 'en'): void { lang = l; }
export function getLang(): 'es' | 'en' { return lang; }
export function t(key: string, params?: Record<string, string | number>): string {
  let s = (lang === 'en' ? en[key] : undefined) ?? es[key] ?? key;
  if (params) for (const k in params) s = s.replace(`{${k}}`, String(params[k]));
  return s;
}
export const MONTHS_ES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic'];
export const MONTHS_EN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export function monthName(i: number): string { return (lang === 'en' ? MONTHS_EN : MONTHS_ES)[((i % 12) + 12) % 12]; }
