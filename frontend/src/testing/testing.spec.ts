import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpTestingController } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import type { EChartsOption } from 'echarts';

import {
  configureTestBed,
  defaultStubFor,
  drainRequests,
  flushMatching,
  getLegend,
  getSeriesAt,
  getTooltip,
  injectHttp,
  legendType,
  seriesLabelShow,
} from './index';

@Component({ template: '<p>ok</p>' })
class TrivialComponent {}

function sampleOption(): EChartsOption {
  return {
    series: [{ label: { show: true } }],
    legend: { type: 'scroll' },
    tooltip: { formatter: '{b}: {d}%' },
  } as unknown as EChartsOption;
}

describe('configureTestBed', () => {
  it('when a trivial standalone component is configured, it mounts zonelessly', async () => {
    await configureTestBed({ imports: [TrivialComponent], withHttp: false });
    const fixture = TestBed.createComponent(TrivialComponent);
    fixture.detectChanges();
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when withHttp defaults to true, injectHttp returns a controller', async () => {
    await configureTestBed({ imports: [TrivialComponent] });
    const http = injectHttp();
    expect(http).toBeTruthy();
    expect(typeof http.verify).toBe('function');
    expect(typeof http.match).toBe('function');
  });
});

describe('http-helpers', () => {
  let http: HttpTestingController;
  let client: HttpClient;

  beforeEach(async () => {
    await configureTestBed({ imports: [TrivialComponent] });
    http = injectHttp();
    client = TestBed.inject(HttpClient);
  });

  afterEach(() => http.verify());

  it('when defaultStubFor sees a known fragment, the matching stub is returned', () => {
    expect(defaultStubFor('/x/market/indices')).toEqual({ indices: [], total: 0 });
    expect(defaultStubFor('/x/unknown')).toEqual({});
  });

  it('when flushMatching is called, matching requests are flushed and counted', () => {
    client.get('/api/rolling-metrics').subscribe();
    const count = flushMatching(http, '/rolling-metrics', { window: 63 });
    expect(count).toBe(1);
  });

  it('when drainRequests is called, all pending requests are flushed', () => {
    client.get('/api/market/indices').subscribe();
    client.get('/api/equity-curve').subscribe();
    drainRequests(http);
    expect(() => http.verify()).not.toThrow();
  });
});

describe('echarts-helpers', () => {
  it('when getSeriesAt is called, the series at the index is returned', () => {
    expect(getSeriesAt(sampleOption())).toEqual({ label: { show: true } });
  });

  it('when seriesLabelShow is called, the label show flag is returned', () => {
    expect(seriesLabelShow(sampleOption())).toBe(true);
  });

  it('when legendType is called, the legend type is returned', () => {
    expect(legendType(sampleOption())).toBe('scroll');
  });

  it('when getLegend and getTooltip are called, their option records are returned', () => {
    expect(getLegend(sampleOption())['type']).toBe('scroll');
    expect(getTooltip(sampleOption())['formatter']).toBe('{b}: {d}%');
  });
});
