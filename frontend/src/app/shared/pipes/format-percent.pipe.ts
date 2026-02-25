import { Pipe, PipeTransform, inject } from '@angular/core';
import { FormatService } from '../../services/format.service';

@Pipe({ name: 'formatPercent', pure: true })
export class FormatPercentPipe implements PipeTransform {
  private readonly fmt = inject(FormatService);

  transform(value: number | null | undefined, decimals?: number): string {
    return this.fmt.formatPercent(value, decimals);
  }
}
