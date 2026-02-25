import { Pipe, PipeTransform, inject } from '@angular/core';
import { FormatService } from '../../services/format.service';

@Pipe({ name: 'formatCurrency', pure: true })
export class FormatCurrencyPipe implements PipeTransform {
  private readonly fmt = inject(FormatService);

  transform(value: number | null | undefined, currency?: string): string {
    return this.fmt.formatCurrency(value, currency);
  }
}
