window.gradioObserver = null;

async function sleep(ms) {
  return new Promise((resolve) => { setTimeout(resolve, ms); });
}

function gradioApp() {
  const elems = document.getElementsByTagName('gradio-app');
  const elem = elems.length === 0 ? document : elems[0];
  if (elem !== document) elem.getElementById = (id) => document.getElementById(id);
  return elem.shadowRoot ? elem.shadowRoot : elem;
}

function logFn(func) { // not recommended: use log, debug or error explicitly
  return async function () { // eslint-disable-line func-names
    const t0 = performance.now();
    const returnValue = func(...arguments);
    const t1 = performance.now();
    log(func.name, `time=${Math.round(t1 - t0)}`);
    timer(func.name, t1 - t0);
    return returnValue;
  };
}

function getUICurrentTab() {
  return gradioApp().querySelector('#tabs button.selected');
}

function getUICurrentTabContent() {
  return gradioApp().querySelector('.tabitem[id^=tab_]:not([style*="display: none"])');
}

const get_uiCurrentTabContent = getUICurrentTabContent;
const get_uiCurrentTab = getUICurrentTab;
const uiAfterUpdateCallbacks = [];
const uiUpdateCallbacks = [];
const uiLoadedCallbacks = [];
const uiReadyCallbacks = [];
const uiTabChangeCallbacks = [];
const optionsChangedCallbacks = [];
let uiCurrentTab = null;
let uiAfterUpdateTimeout = null;

function registerCallback(queue, callback) {
  if (queue.includes(callback)) return;
  queue.push(callback);
}

function onAfterUiUpdate(callback) {
  if (typeof callback !== 'function') {
    error(`onAfterUiUpdate was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiAfterUpdateCallbacks, callback);
}

function onUiUpdate(callback) {
  if (typeof callback !== 'function') {
    error(`onUiUpdate was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiUpdateCallbacks, callback);
}

function onUiLoaded(callback) {
  if (typeof callback !== 'function') {
    error(`onUiLoaded was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiLoadedCallbacks, callback);
}

function onUiReady(callback) {
  if (typeof callback !== 'function') {
    error(`onUiReady was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiReadyCallbacks, callback);
}

function onUiTabChange(callback) {
  if (typeof callback !== 'function') {
    error(`onUiTabChange was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiTabChangeCallbacks, callback);
}

function onOptionsChanged(callback) {
  if (typeof callback !== 'function') {
    error(`onOptionsChanged was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(optionsChangedCallbacks, callback);
}

function executeCallbacks(queue, arg) {
  // if (!uiLoaded) return
  for (const callback of queue) {
    if (!callback) continue;
    try {
      const t0 = performance.now();
      callback(arg);
      const t1 = performance.now();
      if (t1 - t0 > 250) log('callbackSlow', callback.name || callback, `time=${Math.round(t1 - t0)}`);
      timer(callback.name || 'anonymousCallback', t1 - t0);
    } catch (e) {
      error(`executeCallbacks: ${callback} ${e}`);
    }
  }
}

const anyPromptExists = () => gradioApp().querySelectorAll('.main-prompts').length > 0;

function scheduleAfterUiUpdateCallbacks() {
  clearTimeout(uiAfterUpdateTimeout);
  uiAfterUpdateTimeout = setTimeout(() => executeCallbacks(uiAfterUpdateCallbacks), 250);
}

let executedOnLoaded = false;
const ignoreElements = ['logMonitorData', 'logWarnings', 'logErrors', 'tooltip-container', 'logger'];
const ignoreElementsSet = new Set(ignoreElements);
const ignoreClasses = ['wrap'];

let mutationTimer = null;
let validMutations = [];

async function mutationCallback(mutations) {
  if (mutations.length <= 0) return;
  for (const mutation of mutations) {
    const target = mutation.target;
    if (target.nodeName === 'LABEL') continue;
    if (ignoreElementsSet.has(target.id)) continue;
    if (target.classList?.contains(ignoreClasses[0])) continue;
    validMutations.push(mutation);
  }
  if (validMutations.length < 1) return;

  if (mutationTimer) clearTimeout(mutationTimer);
  mutationTimer = setTimeout(async () => {
    if (!executedOnLoaded && anyPromptExists()) { // execute once
      executedOnLoaded = true;
      executeCallbacks(uiLoadedCallbacks);
    }
    if (executedOnLoaded) { // execute on each mutation
      executeCallbacks(uiUpdateCallbacks, mutations);
      scheduleAfterUiUpdateCallbacks();
    }
    const newTab = getUICurrentTab();
    if (newTab && (newTab !== uiCurrentTab)) {
      uiCurrentTab = newTab;
      executeCallbacks(uiTabChangeCallbacks);
    }
    validMutations = [];
    mutationTimer = null;
  }, 100);
}

document.addEventListener('DOMContentLoaded', () => {
  log('DOMContentLoaded');
  window.gradioObserver = new MutationObserver(mutationCallback);
  window.gradioObserver.observe(gradioApp(), { childList: true, subtree: true, attributes: false });
});

/**
 * Add a listener to the document for keydown events
 */
document.addEventListener('keydown', (e) => {
  let elem;
  if (e.key === 'Escape') elem = getUICurrentTabContent().querySelector('button[id$=_interrupt]');
  if (e.key === 'Enter' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_generate]');
  if (e.key === 'i' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_reprocess]');
  if (e.key === ' ' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_extra_networks_btn]');
  if (e.key === 'n' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_extra_networks_btn]');
  if (e.key === 's' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id^=save_]');
  if (e.key === 'Insert' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id^=save_]');
  if (e.key === 'd' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id^=delete_]');
  // if (e.key === 'm' && e.ctrlKey) elem = gradioApp().getElementById('setting_sd_model_checkpoint');
  if (elem) {
    e.preventDefault();
    log('hotkey', { key: e.key, meta: e.metaKey, ctrl: e.ctrlKey, alt: e.altKey }, elem?.id, elem.nodeName);
    if (elem.nodeName === 'BUTTON') elem.click();
    else elem.focus();
  }
});

function getSortableCellValue(cell, sortType) {
  const rawValue = cell?.dataset?.sortValue ?? cell?.textContent?.trim() ?? '';
  if (sortType === 'number') {
    const numericValue = Number.parseFloat(rawValue);
    return Number.isNaN(numericValue) ? Number.NEGATIVE_INFINITY : numericValue;
  }
  return rawValue.toLowerCase();
}

function sortModelListTable(table, columnIndex, sortType, sortOrder) {
  const tbody = table.querySelector('tbody');
  if (!tbody) return;
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const direction = sortOrder === 'desc' ? -1 : 1;
  const sortedRows = rows
    .map((row, index) => ({ row, index }))
    .sort((a, b) => {
      const aCell = a.row.children[columnIndex];
      const bCell = b.row.children[columnIndex];
      const aValue = getSortableCellValue(aCell, sortType);
      const bValue = getSortableCellValue(bCell, sortType);
      if (aValue < bValue) return -1 * direction;
      if (aValue > bValue) return 1 * direction;
      return a.index - b.index;
    });
  tbody.replaceChildren(...sortedRows.map((item) => item.row));
}

function applySortIndicators(table, activeHeader, sortOrder) {
  const headers = table.querySelectorAll('th.sortable');
  for (const header of headers) {
    header.classList.remove('sorted-asc', 'sorted-desc');
    header.removeAttribute('aria-sort');
  }
  activeHeader.classList.add(sortOrder === 'desc' ? 'sorted-desc' : 'sorted-asc');
  activeHeader.setAttribute('aria-sort', sortOrder === 'desc' ? 'descending' : 'ascending');
}

async function initTableSorter() {
  const t0 = performance.now();
  const root = gradioApp();
  for (const table of root.querySelectorAll('table[data-sortable="true"]')) {
    console.log('HERE', table);
    if (!table || table.dataset.sortBound === 'true') return;
    const headers = Array.from(table.querySelectorAll('th.sortable'));
    if (headers.length === 0) return;

    for (const [index, header] of headers.entries()) {
      header.style.cursor = 'pointer';
      header.addEventListener('click', () => {
        const isCurrentHeader = table.dataset.sortKey === header.dataset.sortKey;
        const nextOrder = isCurrentHeader && table.dataset.sortOrder === 'asc' ? 'desc' : 'asc';
        table.dataset.sortKey = header.dataset.sortKey;
        table.dataset.sortOrder = nextOrder;
        sortModelListTable(table, index, header.dataset.sortType || 'text', nextOrder);
        applySortIndicators(table, header, nextOrder);
      });
    }

    const defaultSortKey = table.dataset.defaultSortKey || 'name';
    const defaultSortOrder = table.dataset.defaultSortOrder || 'asc';
    const defaultHeader = headers.find((header) => header.dataset.sortKey === defaultSortKey) || headers[0];
    const defaultIndex = headers.indexOf(defaultHeader);
    table.dataset.sortKey = defaultHeader.dataset.sortKey;
    table.dataset.sortOrder = defaultSortOrder;
    sortModelListTable(table, defaultIndex, defaultHeader.dataset.sortType || 'text', defaultSortOrder);
    applySortIndicators(table, defaultHeader, defaultSortOrder);
    table.dataset.sortBound = 'true';
  }
  onUiUpdate(initTableSorter);
  const t1 = performance.now();
  log('initTableSorter', Math.round(t1 - t0));
  timer('initTableSorter', t1 - t0);
}

async function deleteFile(filename) {
  if (!filename) return;
  if (!confirm(`Are you sure you want to delete the object? This action cannot be undone. ${filename}`)) return; // eslint-disable-line no-alert
  const res = await authFetch(`${window.api}/delete-file?file=${encodeURIComponent(filename)}`);
  if (!res || res.status !== 200) {
    error('FileDelete', { file: filename, status: res?.status, statusText: res?.statusText });
    return;
  }
  const data = await res.json();
  log('FileDelete', data);
}

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
  if (el === document) return true;
  const computedStyle = getComputedStyle(el);
  const isVisible = computedStyle.display !== 'none';
  if (!isVisible) return false;
  return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
  const clRect = el.getBoundingClientRect();
  const windowHeight = window.innerHeight;
  const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;
  return isOnScreen;
}
