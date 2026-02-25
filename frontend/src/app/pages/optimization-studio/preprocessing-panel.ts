import { Component, signal, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HelpTooltipComponent } from '../../shared/help-tooltip/help-tooltip';

type ReturnsType = 'arithmetic' | 'log';
type Frequency = 'daily' | 'weekly' | 'monthly';
type MissingData = 'drop' | 'forward_fill' | 'interpolate';
type OutlierTreatment = 'winsorize' | 'trim' | 'none';

@Component({
  selector: 'app-preprocessing-panel',
  imports: [FormsModule, HelpTooltipComponent],
  templateUrl: './preprocessing-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PreprocessingPanelComponent {
  returnsType = signal<ReturnsType>('arithmetic');
  frequency = signal<Frequency>('daily');
  lookbackYears = signal<number>(5);
  missingData = signal<MissingData>('forward_fill');
  outlierTreatment = signal<OutlierTreatment>('winsorize');

  setReturnsType(value: ReturnsType): void {
    this.returnsType.set(value);
  }

  setFrequency(value: string): void {
    this.frequency.set(value as Frequency);
  }

  setMissingData(value: string): void {
    this.missingData.set(value as MissingData);
  }

  setOutlierTreatment(value: string): void {
    this.outlierTreatment.set(value as OutlierTreatment);
  }

  setLookbackYears(value: number): void {
    this.lookbackYears.set(value);
  }
}
