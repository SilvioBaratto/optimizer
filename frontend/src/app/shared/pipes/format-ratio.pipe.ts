import { Pipe, PipeTransform, inject } from '@angular/core';
import { FormatService } from '../../services/format.service';

@Pipe({ name: 'formatRatio', pure: true })
export class FormatRatioPipe implements PipeTransform {
  private readonly fmt = inject(FormatService);

  transform(value: number | null | undefined, decimals?: number): string {
    return this.fmt.formatRatio(value, decimals);
  }
}
